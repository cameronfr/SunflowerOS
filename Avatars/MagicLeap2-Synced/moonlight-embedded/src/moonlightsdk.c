/*
 * This file is part of Moonlight Embedded.
 *
 * Copyright (C) 2015-2019 Iwan Timmer
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

#include "moonlightsdk.h"

#include "connection.h"
#include "config.h"

#include "audio.h"
#include "video.h"

#include "input/mapping.h"
#ifdef HAVE_LIBCEC
#include "input/cec.h"
#endif
#ifdef HAVE_SDL
#include "input/sdl.h"
#endif

#include <Limelight.h>

#include <client.h>
#include <discover.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <openssl/rand.h>

#include <libavcodec/avcodec.h>


static CONFIGURATION config;
static SERVER_DATA server;

static void applist(PSERVER_DATA server) {
  PAPP_LIST list = NULL;
  if (gs_applist(server, &list) != GS_OK) {
    fprintf(stderr, "Can't get app list\n");
    return;
  }

  for (int i = 1;list != NULL;i++) {
    printf("%d. %s\n", i, list->name);
    list = list->next;
  }
}

static int get_app_id(PSERVER_DATA server, const char *name) {
  PAPP_LIST list = NULL;
  if (gs_applist(server, &list) != GS_OK) {
    fprintf(stderr, "Can't get app list\n");
    return -1;
  }

  while (list != NULL) {
    if (strcmp(list->name, name) == 0)
      return list->id;

    list = list->next;
  }
  return -1;
}

void onFrame(AVFrame *frame) {
  // printf("New frame has format YUV? %d\n", frame->format == AV_PIX_FMT_YUV420P);
}

static void help() {
  printf("Moonlight Embedded Fork of f021439d1bb33b4869273f7521ec77edb6804fe1\n");
  printf("Usage: moonlight [action] (options) [host]\n");
  printf("       moonlight [configfile]\n");
  printf("\n Actions\n\n");
  printf("\tpair\t\t\tPair device with computer\n");
  printf("\tunpair\t\t\tUnpair device with computer\n");
  printf("\tstream\t\t\tStream computer to device\n");
  printf("\tlist\t\t\tList available games and applications\n");
  printf("\tquit\t\t\tQuit the application or game being streamed\n");
  printf("\tmap\t\t\tCreate mapping for gamepad\n");
  printf("\thelp\t\t\tShow this help\n");
  printf("\n Global Options\n\n");
  printf("\t-config <config>\tLoad configuration file\n");
  printf("\t-save <config>\t\tSave configuration file\n");
  printf("\t-verbose\t\tEnable verbose output\n");
  printf("\t-debug\t\t\tEnable verbose and debug output\n");
  printf("\n Streaming options\n\n");
  printf("\t-720\t\t\tUse 1280x720 resolution [default]\n");
  printf("\t-1080\t\t\tUse 1920x1080 resolution\n");
  printf("\t-4k\t\t\tUse 3840x2160 resolution\n");
  printf("\t-width <width>\t\tHorizontal resolution (default 1280)\n");
  printf("\t-height <height>\tVertical resolution (default 720)\n");
  #if defined(HAVE_PI) | defined(HAVE_MMAL)
  printf("\t-rotate <angle>\tRotate display: 0/90/180/270 (default 0)\n");
  #endif
  printf("\t-fps <fps>\t\tSpecify the fps to use (default 60)\n");
  printf("\t-bitrate <bitrate>\tSpecify the bitrate in Kbps\n");
  printf("\t-packetsize <size>\tSpecify the maximum packetsize in bytes\n");
  printf("\t-codec <codec>\t\tSelect used codec: auto/h264/h265 (default auto)\n");
  printf("\t-remote <yes/no/auto>\t\t\tEnable optimizations for WAN streaming (default auto)\n");
  printf("\t-app <app>\t\tName of app to stream\n");
  printf("\t-nosops\t\t\tDon't allow GFE to modify game settings\n");
  printf("\t-localaudio\t\tPlay audio locally on the host computer\n");
  printf("\t-surround <5.1/7.1>\t\tStream 5.1 or 7.1 surround sound\n");
  printf("\t-keydir <directory>\tLoad encryption keys from directory\n");
  printf("\t-mapping <file>\t\tUse <file> as gamepad mappings configuration file\n");
  printf("\t-platform <system>\tSpecify system used for audio, video and input: pi/imx/aml/rk/x11/x11_vdpau/sdl/fake (default auto)\n");
  printf("\t-nounsupported\t\tDon't stream if resolution is not officially supported by the server\n");
  printf("\t-quitappafter\t\tSend quit app request to remote after quitting session\n");
  printf("\t-viewonly\t\tDisable all input processing (view-only mode)\n");
  printf("\t-nomouseemulation\t\tDisable gamepad mouse emulation support (long pressing Start button)\n");
  #if defined(HAVE_SDL) || defined(HAVE_X11)
  printf("\n WM options (SDL and X11 only)\n\n");
  printf("\t-windowed\t\tDisplay screen in a window\n");
  #endif
  #ifdef HAVE_EMBEDDED
  printf("\n I/O options (Not for SDL)\n\n");
  printf("\t-input <device>\t\tUse <device> as input. Can be used multiple times\n");
  printf("\t-audio <device>\t\tUse <device> as audio output device\n");
  #endif
  printf("\nUse Ctrl+Alt+Shift+Q or Play+Back+LeftShoulder+RightShoulder to exit streaming session\n\n");
  exit(0);
}

static void pair_check(PSERVER_DATA server) {
  if (!server->paired) {
    fprintf(stderr, "You must pair with the PC first\n");
    exit(-1);
  }
}

extern bool moon_init_config(char *address, char *app) {
  // Sets to default values, also sets a key dir.
  config_parse(0, NULL, &config);
  config.address = address;
  config.stream.width = 1920;
  config.stream.height = 1080;
  config.app = app;
  // config.address = "192.168.0.4";
  return true;
}

// Depends on config object
extern bool moon_init_server_connection() {
  printf("Connecting to %s, using key_dir %s\n", config.address, config.key_dir);

  int ret;
  if ((ret = gs_init(&server, config.address, config.port, config.key_dir, config.debug_level, config.unsupported)) == GS_OUT_OF_MEMORY) {
    fprintf(stderr, "Not enough memory\n");
    return false;
  } else if (ret == GS_ERROR) {
    fprintf(stderr, "Gamestream error: %s\n", gs_error);
    return false;
  } else if (ret == GS_INVALID) {
    fprintf(stderr, "Invalid data received from server: %s\n", gs_error);
    return false;
  } else if (ret == GS_UNSUPPORTED_VERSION) {
    fprintf(stderr, "Unsupported version: %s\n", gs_error);
    return false;
  } else if (ret != GS_OK) {
    fprintf(stderr, "Can't connect to server %s\n", config.address);
    return false;
  }

  printf("Connected to %s", config.address);
  return true;
}

// Depends on config object and server object
extern bool moon_pair_server() {
  // TODO: I think this blocks main thread
  char pin[5];
  if (config.pin > 0 && config.pin <= 9999) {
    sprintf(pin, "%04d", config.pin);
  } else {
    sprintf(pin, "%d%d%d%d", (int)random() % 10, (int)random() % 10, (int)random() % 10, (int)random() % 10);
  }
  printf("Please enter the following PIN on the target PC: %s\n", pin);
  fflush(stdout);
  if (gs_pair(&server, &pin[0]) != GS_OK) {
    fprintf(stderr, "Failed to pair to server: %s\n", gs_error);
    return false;
  } else {
    printf("Succesfully paired\n");
    return true;
  }
}

// Depends on config object and server object
extern bool moon_start_stream(/*void (*on_frame)(AVFrame*)*/void* on_frame) {
  printf("Starting stream...\n");
  if(!server.paired) {
    fprintf(stderr, "You must pair with the PC first\n");
    return false;
  }

  int appId = get_app_id(&server, config.app);
  if (appId<0) {
    fprintf(stderr, "Can't find app %s\n", config.app);
    return false;
  }

  int gamepads = 0;
  //gamepads += evdev_gamepads;
  // #ifdef HAVE_SDL
  // gamepads += sdl_gamepads;
  // #endif
  int gamepad_mask = 0;
  for (int i = 0; i < gamepads && i < 4; i++)
    gamepad_mask = (gamepad_mask << 1) + 1;

  int ret = gs_start_app(&server, &config.stream, appId, config.sops, config.localaudio, gamepad_mask);
  if (ret < 0) {
    if (ret == GS_NOT_SUPPORTED_4K)
      fprintf(stderr, "Server doesn't support 4K\n");
    else if (ret == GS_NOT_SUPPORTED_MODE)
      fprintf(stderr, "Server doesn't support %dx%d (%d fps) or remove --nounsupported option\n", config.stream.width, config.stream.height, config.stream.fps);
    else if (ret == GS_NOT_SUPPORTED_SOPS_RESOLUTION)
      fprintf(stderr, "Optimal Playable Settings isn't supported for the resolution %dx%d, use supported resolution or add --nosops option\n", config.stream.width, config.stream.height);
    else if (ret == GS_ERROR)
      fprintf(stderr, "Gamestream error: %s\n", gs_error);
    else
      fprintf(stderr, "Errorcode starting app: %d\n", ret);
    return false;
  }

  int drFlags = 0;
  if (config.fullscreen)
    drFlags |= DISPLAY_FULLSCREEN;

  switch (config.rotate) {
  case 0:
    break;
  case 90:
    drFlags |= DISPLAY_ROTATE_90;
    break;
  case 180:
    drFlags |= DISPLAY_ROTATE_180;
    break;
  case 270:
    drFlags |= DISPLAY_ROTATE_270;
    break;
  default:
    printf("Ignoring invalid rotation value: %d\n", config.rotate);
  }

  if (config.debug_level > 0) {
    printf("Stream %d x %d, %d fps, %d kbps\n", config.stream.width, config.stream.height, config.stream.fps, config.stream.bitrate);
    connection_debug = true;
  }

  LiStartConnection(&server.serverInfo, &config.stream, &connection_callbacks, &video_decoder_callbacks, &audio_decoder_callbacks, on_frame, drFlags, config.audio_device, 0);
  return true;
}

extern void moon_stop_stream() {
  LiStopConnection();

  // if (config.quitappafter) {
  //   if (config.debug_level > 0)
  //     printf("Sending app quit request ...\n");
  //   gs_quit_app(server);
  // }
}

int main(int argc, char* argv[]) {
  config_parse(argc, argv, &config);

  if (config.action == NULL || strcmp("help", config.action) == 0)
    help();

  if (strcmp("map", config.action) == 0) {
    if (config.inputsCount != 1) {
      printf("You need to specify one input device using -input.\n");
      exit(-1);
    }

    // evdev_create(config.inputs[0], NULL, config.debug_level > 0, config.rotate);
    //evdev_map(config.inputs[0]);
    exit(0);
  }

  if (config.address == NULL) {
    config.address = (char*)malloc(MAX_ADDRESS_SIZE);
    if (config.address == NULL) {
      perror("Not enough memory");
      exit(-1);
    }
    config.address[0] = 0;
    printf("Searching for server...\n");
    gs_discover_server(config.address, &config.port);
    if (config.address[0] == 0) {
      fprintf(stderr, "Autodiscovery failed. Specify an IP address next time.\n");
      exit(-1);
    }
  }

  char host_config_file[128];
  sprintf(host_config_file, "hosts/%s.conf", config.address);
  if (access(host_config_file, R_OK) != -1)
    config_file_parse(host_config_file, &config);

  moon_init_server_connection();
  printf("paired? %d\n", server.paired);

  if (config.debug_level > 0)
    printf("GPU: %s, GFE: %s (%s, %s)\n", server.gpuType, server.serverInfo.serverInfoGfeVersion, server.gsVersion, server.serverInfo.serverInfoAppVersion);

  if (strcmp("list", config.action) == 0) {
    pair_check(&server);
    applist(&server);
  } else if (strcmp("stream", config.action) == 0) {
    pair_check(&server);

    config.stream.supportsHevc = false;//config.codec != CODEC_H264 && (config.codec == CODEC_HEVC || platform_supports_hevc(system));

    // sdl_init(config.stream.width, config.stream.height, config.fullscreen);

    if (config.viewonly) {
      if (config.debug_level > 0)
        printf("View-only mode enabled, no input will be sent to the host computer\n");
    } else {
        // sdlinput_init(config.mapping);
    }

    moon_start_stream((void*)&onFrame);
    while(true) {}
  } else if (strcmp("pair", config.action) == 0) {
    moon_pair_server();
  } else if (strcmp("unpair", config.action) == 0) {
    if (gs_unpair(&server) != GS_OK) {
      fprintf(stderr, "Failed to unpair to server: %s\n", gs_error);
    } else {
      printf("Succesfully unpaired\n");
    }
  } else if (strcmp("quit", config.action) == 0) {
    pair_check(&server);
    gs_quit_app(&server);
  } else
    fprintf(stderr, "%s is not a valid action\n", config.action);
}