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
#include "ffmpeg.h"

#include <unistd.h>
#include <stdbool.h>

#define DECODER_BUFFER_SIZE 256*1024
#define SLICES_PER_FRAME 4

static char* ffmpeg_buffer;
// define on_frame_received function pointer
void (*on_frame_received)(AVFrame *frame);

static int setup(int videoFormat, int width, int height, int redrawRate, void* context, int drFlags) {
  on_frame_received = context;

  if (ffmpeg_init(videoFormat, width, height, SLICE_THREADING, 2, SLICES_PER_FRAME) < 0) {
    fprintf(stderr, "Couldn't initialize video decoding\n");
    return -1;
  }

  ffmpeg_buffer = malloc(DECODER_BUFFER_SIZE + AV_INPUT_BUFFER_PADDING_SIZE);
  if (ffmpeg_buffer == NULL) {
    fprintf(stderr, "Not enough memory\n");
    ffmpeg_destroy();
    return -1;
  }

  return 0;
}

static void cleanup() {
  ffmpeg_destroy();
}

static int submit_decode_unit(PDECODE_UNIT decodeUnit) {
  if (decodeUnit->fullLength < DECODER_BUFFER_SIZE) {
    PLENTRY entry = decodeUnit->bufferList;
    int length = 0;
    while (entry != NULL) {
      memcpy(ffmpeg_buffer+length, entry->data, entry->length);
      length += entry->length;
      entry = entry->next;
    }
    ffmpeg_decode(ffmpeg_buffer, length);

    // if (SDL_LockMutex(mutex) == 0) {
      AVFrame* frame = ffmpeg_get_frame(false);
      if (frame != NULL) {
        on_frame_received(frame);
      }

      // SDL_UnlockMutex(mutex);
    // } else
      // fprintf(stderr, "Couldn't lock mutex\n");
  } else {
    fprintf(stderr, "Video decode buffer too small");
    exit(1);
  }
  return DR_OK;
}

DECODER_RENDERER_CALLBACKS video_decoder_callbacks = {
  .setup = setup,
  .cleanup = cleanup,
  .submitDecodeUnit = submit_decode_unit,
  .capabilities = CAPABILITY_SLICES_PER_FRAME(SLICES_PER_FRAME) | CAPABILITY_REFERENCE_FRAME_INVALIDATION_HEVC | CAPABILITY_DIRECT_SUBMIT,
};

