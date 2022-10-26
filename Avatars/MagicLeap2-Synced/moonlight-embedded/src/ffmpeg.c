/*
 * This file is part of Moonlight Embedded.
 *
 * Based on Moonlight Pc implementation
 *
 * Moonlight is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Moonlight is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Moonlight; if not, see <http://www.gnu.org/licenses/>.
 */

#include "ffmpeg.h"

#ifdef HAVE_VAAPI
#include "ffmpeg_vaapi.h"
#endif

#include <Limelight.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>

#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>
#include <stdbool.h>

#include <jni.h>

//JNI_Onload
jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
  printf("JNI_OnLoad called, setting libavcodec vm;");
  av_jni_set_java_vm(vm, NULL);
  return JNI_VERSION_1_6;
}


// General decoder and renderer state
static AVPacket* pkt;
static const AVCodec* decoder;
static AVCodecContext* decoder_ctx;
static AVFrame** dec_frames;

static int dec_frames_cnt;
static int current_frame, next_frame;

enum decoders ffmpeg_decoder;

// For converting to rgb
static AVFrame *rgb_frame;
static struct SwsContext *sws_ctx;

#define BYTES_PER_PIXEL 4



// This function must be called before
// any other decoding functions
int ffmpeg_init(int videoFormat, int width, int height, int perf_lvl, int buffer_count, int thread_count) {
  // Initialize the avcodec library and register codecs
  av_log_set_level(AV_LOG_QUIET);
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58,10,100)
  avcodec_register_all();
#endif


  // av_log_set_level(AV_LOG_DEBUG);
  pkt = av_packet_alloc();
  if (pkt == NULL) {
    printf("Couldn't allocate packet\n");
    return -1;
  }

  ffmpeg_decoder = perf_lvl & VAAPI_ACCELERATION ? VAAPI : SOFTWARE;
  switch (videoFormat) {
    case VIDEO_FORMAT_H264:
      // decoder = avcodec_find_decoder_by_name("h264_mediacodec");
      decoder = avcodec_find_decoder_by_name("h264");
      break;
    case VIDEO_FORMAT_H265:
      // decoder = avcodec_find_decoder_by_name("hevc_mediacodec");
      decoder = avcodec_find_decoder_by_name("hevc");
      break;
  }

  if (decoder == NULL) {
    printf("Couldn't find decoder\n");
    printf("Was looking for h264_mediacodec/hevc_mediacodec specifically, make sure ffmpeg is built with it\n");
    return -1;
  }

  decoder_ctx = avcodec_alloc_context3(decoder);
  if (decoder_ctx == NULL) {
    printf("Couldn't allocate context");
    return -1;
  }

  // Use low delay decoding
  decoder_ctx->flags |= AV_CODEC_FLAG_LOW_DELAY;

  // Allow display of corrupt frames and frames missing references
  decoder_ctx->flags |= AV_CODEC_FLAG_OUTPUT_CORRUPT;
  decoder_ctx->flags2 |= AV_CODEC_FLAG2_SHOW_ALL;

  // Report decoding errors to allow us to request a key frame
  decoder_ctx->err_recognition = AV_EF_EXPLODE;

  if (perf_lvl & SLICE_THREADING) {
    decoder_ctx->thread_type = FF_THREAD_SLICE;
    decoder_ctx->thread_count = thread_count;
  } else {
    decoder_ctx->thread_count = 1;
  }

  decoder_ctx->width = width;
  decoder_ctx->height = height;
  decoder_ctx->pix_fmt = AV_PIX_FMT_YUV420P;

  dec_frames_cnt = buffer_count;
  dec_frames = (AVFrame **)malloc(buffer_count * sizeof(AVFrame*));
  if (dec_frames == NULL) {
    fprintf(stderr, "Couldn't allocate frames");
    return -1;
  }

  for (int i = 0; i < buffer_count; i++) {
    dec_frames[i] = av_frame_alloc();
    if (dec_frames[i] == NULL) {
      fprintf(stderr, "Couldn't allocate frame");
      return -1;
    }
  }

  rgb_frame = av_frame_alloc();
  rgb_frame->width = width;
  rgb_frame->height = height;
  rgb_frame->format = AV_PIX_FMT_RGB24;
  sws_ctx = sws_getContext(width, height, AV_PIX_FMT_YUV420P, width, height, AV_PIX_FMT_RGB24, SWS_FAST_BILINEAR, NULL, NULL, NULL);

  return 0;
}

// This function must be called after
// decoding is finished
void ffmpeg_destroy(void) {
  av_packet_free(&pkt);
  if (decoder_ctx) {
    avcodec_close(decoder_ctx);
    av_free(decoder_ctx);
    decoder_ctx = NULL;
  }
  if (dec_frames) {
    for (int i = 0; i < dec_frames_cnt; i++) {
      if (dec_frames[i])
        av_frame_free(&dec_frames[i]);
    }
  }
  av_frame_free(&rgb_frame);
  sws_freeContext(sws_ctx);
}

AVFrame* ffmpeg_get_frame(bool native_frame) {
  int err = avcodec_receive_frame(decoder_ctx, dec_frames[next_frame]);
  if (err == 0) {
    current_frame = next_frame;
    next_frame = (current_frame+1) % dec_frames_cnt;

    if (ffmpeg_decoder == SOFTWARE || native_frame) {

      AVFrame *frame_yuv = dec_frames[current_frame];
      // sws_scale(sws_ctx, frame_yuv->data, frame_yuv->linesize, 0, frame_yuv->height, rgb_frame->data, rgb_frame->linesize);
      // return rgb_frame;
      return dec_frames[current_frame];
    }
  } else if (err != AVERROR(EAGAIN)) {
    char errorstring[512];
    av_strerror(err, errorstring, sizeof(errorstring));
    // fprintf(stderr, "Receive failed - %d/%s\n", err, errorstring);
  }
  return NULL;
}

void ffmpeg_submit_sps_pps(char *sps, int sps_len, char *pps, int pps_len) {
  // sps, pps are already NALU data
  if (decoder_ctx->extradata == NULL) {
    unsigned char *extradata = (unsigned char*) malloc(sps_len + pps_len);
    memcpy(extradata, sps, sps_len);
    memcpy(extradata + sps_len, pps, pps_len);
    decoder_ctx->extradata = extradata;
    decoder_ctx->extradata_size = sps_len + pps_len;
    printf("Submitted SPS/PPS to mediacodec decoder\n");
    printf("sps first 4 bytes: %02x %02x %02x %02x\n", sps[0], sps[1], sps[2], sps[3]);
    printf("pps first 4 bytes: %02x %02x %02x %02x\n", pps[0], pps[1], pps[2], pps[3]);

    int err = avcodec_open2(decoder_ctx, decoder, NULL);
    if (err < 0) {
      char errorstring[512];
      av_strerror(err, errorstring, sizeof(errorstring));
      printf("Couldn't open codec2 - %d/%s\n", err, errorstring);
    } else {
      fprintf(stderr,"Opened mediacodec decoder\n");
    }
  }
}

// packets must be decoded in order
// indata must be inlen + AV_INPUT_BUFFER_PADDING_SIZE in length
int ffmpeg_decode(unsigned char* indata, int inlen) {
  int err;

  pkt->data = indata;
  pkt->size = inlen;

  if (decoder_ctx->extradata != NULL) {
    err = avcodec_send_packet(decoder_ctx, pkt);
    if (err < 0) {
      char errorstring[512];
      av_strerror(err, errorstring, sizeof(errorstring));
      fprintf(stderr, "Decode failed - %s\n", errorstring);
    }
    return err < 0 ? err : 0;
  }

  printf("No decoder opened yet\n");
  return -1;
}
