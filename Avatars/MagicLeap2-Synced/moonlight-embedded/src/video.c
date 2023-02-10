/*
 * This file is part of Moonlight Embedded.
 *
 * Copyright (C) 2015 Iwan Timmer
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

#include "video.h"
#include <jni.h>

#include <unistd.h>
#include <stdbool.h>
#include <libyuv.h>

// Want sort of UI here:
// 1. Estimated rount-trip time
// 2. Estimated decode time
// 3. Time between decode and display on sk
// 4. Ideally, also encoder time

#define SLICES_PER_FRAME 1 //TODO: try adj this

// define on_frame_received function pointer
void (*on_frame_received)(uint8_t *frame, int width, int height);
// extern "C" JavaVM *java_vm;

jobject codec;
jobject next_input_buffer;
uint8_t* next_input_buffer_data;
int next_input_buffer_position;
int next_input_buffer_size;

int video_width;
int video_height;
int output_yuv_stride;
int output_color_format;

jint next_input_buffer_index;
bool needsCSD;
const int BUFFER_FLAG_CODEC_CONFIG = 2;
const int BUFFER_FLAG_END_OF_STREAM = 4;
const int BUFFER_FLAG_KEY_FRAME = 1;
const int BUFFER_FLAG_PARTIAL_FRAME = 8;
long lastInputBufferTimestampUs = -1;

bool exceptionCheck(JNIEnv *env) {
  if (env->ExceptionCheck()) {
      env->ExceptionDescribe();
      env->ExceptionClear();
      return true;
  }
  return false;
}

JNIEnv* jni_get_env() {
  if (java_vm == NULL) {
    fprintf(stderr, "JavaVM java_vm is NULL\n");
    return NULL;
  }

  JNIEnv *env = NULL;
  int getEnvStat = java_vm->GetEnv((void **)&env, JNI_VERSION_1_6);
  switch(getEnvStat) {
    case JNI_OK:
      break;
    case JNI_EDETACHED:
      if (java_vm->AttachCurrentThread(&env, NULL) != 0) {
        fprintf(stderr, "Failed to attach\n");
      }    
      break;
    case JNI_EVERSION:
      fprintf(stderr, "JNI version not supported\n");
      break;
    default:
      fprintf(stderr, "Failed to get the environment using GetEnv()\n");
      break;
  }
  if (env == NULL) {
    fprintf(stderr, "Failed to get the environment using GetEnv()\n");
  }

  return env;
}

static int setup(int videoFormat, int width, int height, int redrawRate, void* context, int drFlags) {
  // cast to correct type
  on_frame_received = (void (*)(uint8_t *frame, int width, int height)) context;
  video_width = width;
  video_height = height;

  JNIEnv *env = jni_get_env();

  //  "omx.mesa.video_decoder.avc": 6%
  // "omx.google.h264.decoder": 7%
  // in comparison yuv->rgb with libyuv uses ~10% cpu

  // also don't notice latency difference between these two. 
  jclass media_codec = env->FindClass("android/media/MediaCodec");
  jmethodID create_by_codec_name = env->GetStaticMethodID(media_codec, "createByCodecName", "(Ljava/lang/String;)Landroid/media/MediaCodec;");
  jstring codec_name = env->NewStringUTF("OMX.mesa.video_decoder.avc"); //hardware Yuv420pSemiPlanar
  // jstring codec_name = env->NewStringUTF("OMX.google.h264.decoder"); //software Yuv420pPLanar
  jobject local_codec = env->CallStaticObjectMethod(media_codec, create_by_codec_name, codec_name);
  codec = env->NewGlobalRef(local_codec);
  if (exceptionCheck(env)) return -1;

  // configure codec
  jclass media_format = env->FindClass("android/media/MediaFormat");
  jmethodID create_video_format = env->GetStaticMethodID(media_format, "createVideoFormat", "(Ljava/lang/String;II)Landroid/media/MediaFormat;");
  jstring mime = env->NewStringUTF("video/avc");
  jobject format = env->CallStaticObjectMethod(media_format, create_video_format, mime, width, height);
  jmethodID set_integer = env->GetMethodID(media_format, "setInteger", "(Ljava/lang/String;I)V");
  jstring key_low_latency = env->NewStringUTF("low-latency");
  env->CallVoidMethod(format, set_integer, key_low_latency, 1);
  jmethodID configure = env->GetMethodID(media_codec, "configure", "(Landroid/media/MediaFormat;Landroid/view/Surface;Landroid/media/MediaCrypto;I)V");
  env->CallVoidMethod(codec, configure, format, NULL, NULL, 0);

  // start codec
  jmethodID start = env->GetMethodID(media_codec, "start", "()V");
  env->CallVoidMethod(codec, start);

  // get output format
  jmethodID get_output_format = env->GetMethodID(media_codec, "getOutputFormat", "()Landroid/media/MediaFormat;");
  jobject output_format = env->CallObjectMethod(codec, get_output_format);
  jmethodID get_integer = env->GetMethodID(media_format, "getInteger", "(Ljava/lang/String;)I");
  jmethodID contains_key = env->GetMethodID(media_format, "containsKey", "(Ljava/lang/String;)Z");

  jstring key_stride = env->NewStringUTF("stride");
  if (env->CallBooleanMethod(output_format, contains_key, key_stride)) {
    int stride = env->CallIntMethod(output_format, get_integer, key_stride);
    printf("Decoder has stride: %d", stride);
    output_yuv_stride = stride;
  } else {
    fprintf(stderr, "Decoder does not have stride");
    return -1;
  }
  jstring key_color_format = env->NewStringUTF("color-format");
  if (env->CallBooleanMethod(output_format, contains_key, key_color_format)) {
    int color_format = env->CallIntMethod(output_format, get_integer, key_color_format);
    printf("Decoder has color format: %d", color_format);
    output_color_format = color_format;
  } else {
    fprintf(stderr, "Decoder does not have color format");
    return -1;
  }
  jstring key_width = env->NewStringUTF("width");
  if (env->CallBooleanMethod(output_format, contains_key, key_width)) {
    int width = env->CallIntMethod(output_format, get_integer, key_width);
    printf("Decoder has width: %d", width);
  }
  jstring key_height = env->NewStringUTF("height");
  if (env->CallBooleanMethod(output_format, contains_key, key_height)) {
    int height = env->CallIntMethod(output_format, get_integer, key_height);
    printf("Decoder has height: %d", height);
  }
  jstring key_crop_left = env->NewStringUTF("crop-left");
  if (env->CallBooleanMethod(output_format, contains_key, key_crop_left)) {
    int crop_left = env->CallIntMethod(output_format, get_integer, key_crop_left);
    printf("Decoder has crop-left: %d", crop_left);
  }
  jstring key_crop_top = env->NewStringUTF("crop-top");
  if (env->CallBooleanMethod(output_format, contains_key, key_crop_top)) {
    int crop_top = env->CallIntMethod(output_format, get_integer, key_crop_top);
    printf("Decoder has crop-top: %d", crop_top);
  }
  jstring key_crop_right = env->NewStringUTF("crop-right");
  if (env->CallBooleanMethod(output_format, contains_key, key_crop_right)) {
    int crop_right = env->CallIntMethod(output_format, get_integer, key_crop_right);
    printf("Decoder has crop-right: %d", crop_right);
  }
  jstring key_crop_bottom = env->NewStringUTF("crop-bottom");
  if (env->CallBooleanMethod(output_format, contains_key, key_crop_bottom)) {
    int crop_bottom = env->CallIntMethod(output_format, get_integer, key_crop_bottom);
    printf("Decoder has crop-bottom: %d", crop_bottom);
  }



  needsCSD = true;

  printf("Mediacodec initialized!\n");

  return 0;
}

static void cleanup() {
  JNIEnv *env = jni_get_env();
  if (codec != NULL) {
    jmethodID release = env->GetMethodID(env->GetObjectClass(codec), "release", "()V");
    env->CallVoidMethod(codec, release);
    env->DeleteGlobalRef(codec);
    codec = NULL;
  }
  if (next_input_buffer != NULL) {
    env->DeleteGlobalRef(next_input_buffer);
    next_input_buffer = NULL;
  }
}

void fetch_next_input_buffer() {
  JNIEnv *env = jni_get_env();
  if (next_input_buffer != NULL) {
    fprintf(stderr, "next_input_buffer is not NULL, shouln't happen!\n");
    env->DeleteGlobalRef(next_input_buffer);
    next_input_buffer = NULL;
    next_input_buffer_index = -1;
  }
  jmethodID dequeue_input_buffer = env->GetMethodID(env->GetObjectClass(codec), "dequeueInputBuffer", "(J)I");
  // next_input_buffer_index = env->CallIntMethod(codec, dequeue_input_buffer, 10000);
  next_input_buffer_index = env->CallIntMethod(codec, dequeue_input_buffer, 0);
  if (next_input_buffer_index >= 0) {
    jmethodID get_input_buffer = env->GetMethodID(env->GetObjectClass(codec), "getInputBuffer", "(I)Ljava/nio/ByteBuffer;");
    jobject local_next_input_buffer = env->CallObjectMethod(codec, get_input_buffer, next_input_buffer_index);
    next_input_buffer = env->NewGlobalRef(local_next_input_buffer);

    next_input_buffer_data = (uint8_t*) env->GetDirectBufferAddress(next_input_buffer);
    next_input_buffer_size = env->GetDirectBufferCapacity(next_input_buffer);
    next_input_buffer_position = 0;
  } else {
    printf("No input buffer available\n");
    next_input_buffer = NULL;
    next_input_buffer_index = -1;
  }
}

void write_input_buffer(uint8_t* data, int size) {
  JNIEnv *env = jni_get_env();
  if (next_input_buffer == NULL) {
    fprintf(stderr, "next_input_buffer is NULL, shouln't happen!\n");
    return;
  }
  if (next_input_buffer_position + size > next_input_buffer_size) {
    fprintf(stderr, "next_input_buffer is too small!\n");
    return;
  }
  // note sizeof(unsigned char) == 1
  memcpy(next_input_buffer_data + next_input_buffer_position, data, size);
  next_input_buffer_position += size;
}

void submit_input_buffer(long timestampUs, int flags) {
  JNIEnv *env = jni_get_env();
  if (next_input_buffer == NULL) {
    fprintf(stderr, "No input buffer to submit, shouldn't be here!\n!");
    return;
  } 
  jmethodID queue_input_buffer = env->GetMethodID(env->GetObjectClass(codec), "queueInputBuffer", "(IIIJI)V");
  if(exceptionCheck(env)) return;
  if (timestampUs <= lastInputBufferTimestampUs) {
    timestampUs = lastInputBufferTimestampUs + 1;
  }
  env->CallVoidMethod(codec, queue_input_buffer, next_input_buffer_index, 0, next_input_buffer_position, timestampUs, flags);
  lastInputBufferTimestampUs = timestampUs;
  if(exceptionCheck(env)) return;
  // TODO: request IDR frame if there's transient error.Also might need to recover or smthn.
  env->DeleteGlobalRef(next_input_buffer);
  next_input_buffer = NULL;
  next_input_buffer_index = -1;
}

static int submit_decode_unit(PDECODE_UNIT decodeUnit) {
  // Reference is moonlight-android/.../MediaCodecDecoderRenderer.java
  // printf("submit_decode_unit frame#%d:\n", decodeUnit->frameNumber);

  JNIEnv *env = jni_get_env();

  if (needsCSD && decodeUnit->frameType != FRAME_TYPE_IDR) {
    printf("Waiting for initial IDR framee");
    return DR_NEED_IDR;
  }
  if (next_input_buffer == NULL) {
    // on first frame, we need to fetch the input buffer
    fetch_next_input_buffer();
  }

  if (decodeUnit->frameType == FRAME_TYPE_IDR) {
    printf("submit_decode_unut: processing IDR frame\n");

    PLENTRY entry = decodeUnit->bufferList;
    if (entry->bufferType != BUFFER_TYPE_SPS) {
      fprintf(stderr, "Expected SPS, PPS, then PicData\n");
    }
    uint8_t* sps = (uint8_t*)entry->data;
    int spsLen = entry->length;
    entry = entry->next;
    if (entry->bufferType != BUFFER_TYPE_PPS) {
      fprintf(stderr, "Expected SPS, PPS, then PicData\n");
    }
    uint8_t* pps = (uint8_t*)entry->data;
    int ppsLen = entry->length;
    
    if(needsCSD) {
      // Submit CSD. After this runs, we'll submit non-fused IDR frame.
      needsCSD = false;
      write_input_buffer(sps, spsLen);
      write_input_buffer(pps, ppsLen);
      int inputBufferFlags = BUFFER_FLAG_CODEC_CONFIG;
      long inputBufferTimestampUs = 0;
      printf("submit_decode_unit: making CSD buffer, flags are %d, timestamp is %ld\n", inputBufferFlags, inputBufferTimestampUs);
      submit_input_buffer(inputBufferTimestampUs, inputBufferFlags);
      fetch_next_input_buffer();
      printf("submit_decode_unit: CSD buffer submitted, starting non-fused IDR buffer\n");
    } else {
      // Otherwise, build fused IDR frame
      write_input_buffer(sps, spsLen);
      write_input_buffer(pps, ppsLen);
      printf("submit_decode_unit: starting fused IDR buffer\n");
    }
    printf("submit_decode_unit: finishing IDR buffer\n");

    int inputBufferFlags = BUFFER_FLAG_KEY_FRAME;
    long inputBufferTimestampUs = decodeUnit->enqueueTimeMs * 1000;
    entry = entry->next;
    while(entry != NULL) {
      if (entry->bufferType != BUFFER_TYPE_PICDATA) {
        fprintf(stderr, "Expected SPS, PPS, then PicData\n");
      }
      uint8_t* picData = (uint8_t*)entry->data;
      int picDataLen = entry->length;
      write_input_buffer(picData, picDataLen);
      entry = entry->next;
    }
    submit_input_buffer(inputBufferTimestampUs, inputBufferFlags);
    fetch_next_input_buffer();
    printf("submit_decode_unit: submitted IDR input buffer\n");
  } else {
    PLENTRY entry = decodeUnit->bufferList;

    int inputBufferFlags = 0;
    long inputBufferTimestampUs = decodeUnit->enqueueTimeMs * 1000;

    while(entry != NULL) {
      if (entry->bufferType != BUFFER_TYPE_PICDATA) {
        fprintf(stderr, "Expected PicData");
      }

      uint8_t* picdata = (uint8_t*)entry->data;
      int picdataLen = entry->length;
      write_input_buffer(picdata, picdataLen);
      entry = entry->next;
    }
    submit_input_buffer(inputBufferTimestampUs, inputBufferFlags);
    // printf("submit_decode_unit: submitted P frame input buffer\n");
    fetch_next_input_buffer();
  }

  // keep going until we don't get output buffer. In beginning, might build up a bunch of output buffers. basically clear the queue.
  int bufCount = 0;
  while(true){
    // printf("submit_decode_unit: getting output buffer\n");
    jmethodID dequeue_output_buffer = env->GetMethodID(env->GetObjectClass(codec), "dequeueOutputBuffer", "(Landroid/media/MediaCodec$BufferInfo;J)I");
    jmethodID buffer_info_constructor = env->GetMethodID(env->FindClass("android/media/MediaCodec$BufferInfo"), "<init>", "()V");
    jclass buffer_info_class = env->FindClass("android/media/MediaCodec$BufferInfo");
    jobject buffer_info = env->NewObject(buffer_info_class, buffer_info_constructor);
    int output_buffer_index = env->CallIntMethod(codec, dequeue_output_buffer, buffer_info, 8000);
    if(exceptionCheck(env)) return DR_NEED_IDR;
    if (output_buffer_index < 0) {
      if (bufCount == 0) {
        printf("submit_decode_unit: no output buffer on this loop run!\n");
      }
      return DR_OK;
    } else if (output_buffer_index >= 0 && bufCount > 0) {
      printf("submit_decode_unit: got an additional output buffer on this loop run\n", bufCount);
    }
    bufCount ++;
    // printf("submit_decode_unit: got output buffer\n");
    // get output buffer data
    jmethodID get_buffer = env->GetMethodID(env->GetObjectClass(codec), "getOutputBuffer", "(I)Ljava/nio/ByteBuffer;");
    jobject output_buffer = env->CallObjectMethod(codec, get_buffer, output_buffer_index);

    uint8_t *output_buffer_data = (uint8_t*)env->GetDirectBufferAddress(output_buffer);
    int output_buffer_size = env->GetDirectBufferCapacity(output_buffer);
    // printf("submit_decode_unit: got output buffer data\n");


    
    uint8_t* rgbBuf = (uint8_t*)malloc(video_width * video_height * 4);
    if (output_color_format == 21) {
      int ret = libyuv::NV12ToABGR(
        output_buffer_data, 
        output_yuv_stride, 
        output_buffer_data + output_yuv_stride *  video_height + (output_yuv_stride - video_width)/2, 
        output_yuv_stride, 
        rgbBuf, 
        video_width * 4, 
        video_width, 
        video_height);
    } else if (output_color_format == 19) {
      int ret = libyuv::I420ToABGR(
        output_buffer_data, 
        output_yuv_stride, 
        output_buffer_data + output_yuv_stride *  video_height, 
        output_yuv_stride/2, 
        output_buffer_data + output_yuv_stride *  video_height + (output_yuv_stride * video_height)/4, 
        output_yuv_stride/2, 
        rgbBuf, 
        video_width * 4, 
        video_width, 
        video_height);
    } else {
      fprintf(stderr, "Unsupported output color format: %d", output_color_format);
    }

    on_frame_received(rgbBuf, video_width, video_height); // give ownership of rgbBuf to callback (i.e., it's callbacks responsibility to free it 😤)
    

    // release output buffer
    jmethodID release_output_buffer = env->GetMethodID(env->GetObjectClass(codec), "releaseOutputBuffer", "(IZ)V");
    env->CallVoidMethod(codec, release_output_buffer, output_buffer_index, false);
  }
}

DECODER_RENDERER_CALLBACKS video_decoder_callbacks = {
  .setup = setup,
  .cleanup = cleanup,
  .submitDecodeUnit = submit_decode_unit,
  .capabilities = CAPABILITY_SLICES_PER_FRAME(SLICES_PER_FRAME) | CAPABILITY_REFERENCE_FRAME_INVALIDATION_HEVC,// | CAPABILITY_DIRECT_SUBMIT,
};
