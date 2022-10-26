# Can't Unload/Reload header file if it has #pragma once

#%%#

import cppyy
import os

cppyy.cppdef("""
#include <stdio.h>
#include <stdlib.h>
#include <android/log.h>
// fake printf using android log
#define printf(...) __android_log_print(ANDROID_LOG_INFO, "Sunflower cppyy", __VA_ARGS__)
""")

# Set java_vm global variable
cppyy.cppdef("""
#include <jni.h>
JavaVM *java_vm;
jobject java_android_activity;
void get_java_vm_from_env() {
  sscanf(getenv("JAVA_VM_PTR"), "%p", &java_vm);
  printf("java_vm: %p", java_vm);

  JNIEnv *env;
  java_vm->GetEnv((void **)&env, JNI_VERSION_1_6);
  jclass activity_thread = env->FindClass("android/app/ActivityThread");
  jmethodID current_activity_thread = env->GetStaticMethodID(activity_thread, "currentActivityThread", "()Landroid/app/ActivityThread;");
  jobject at = env->CallStaticObjectMethod(activity_thread, current_activity_thread);
  jmethodID get_application = env->GetMethodID(activity_thread, "getApplication", "()Landroid/app/Application;");
  jobject activity_inst = env->CallObjectMethod(at, get_application);
  java_android_activity = env->NewGlobalRef(activity_inst);
}
""")
cppyy.gbl.get_java_vm_from_env()

# Load StereoKit
cppyy.add_include_path(os.path.expanduser("~/MagicLeap2-Synced/StereoKit/StereoKitC"))
cppyy.add_library_path(os.path.expanduser("~/MagicLeap2-Synced/StereoKit/"))
cppyy.load_library("libStereoKitC.so")

# Load Moonlight-Embedded
moonlightEmbeddedPath = os.path.expanduser("~/MagicLeap2-Synced/moonlight-embedded")
systemIncludePath ="/data/data/com.termux/files/usr/include"
cppyy.add_include_path(os.path.join(systemIncludePath, "opus"))
cppyy.add_include_path(os.path.join(moonlightEmbeddedPath, "src"))
cppyy.add_include_path(os.path.join(moonlightEmbeddedPath, "libgamestream"))
cppyy.add_include_path(os.path.join(moonlightEmbeddedPath, ""))
cppyy.add_include_path(os.path.join(moonlightEmbeddedPath, "third_party/moonlight-common-c/src"))
cppyy.add_library_path(os.path.join(moonlightEmbeddedPath, "libgamestream"))

cppyy.add_include_path(os.path.expanduser("~/MagicLeap2-Synced/moonlight-embedded/src/"))
# cppyy.add_library_path(os.path.expanduser("~/MagicLeap2-Synced/moonlight-embedded/"))
# cppyy.load_library("libmoonlight.so")
cppyy.load_library("libswscale.so")
cppyy.load_library("libavcodec.so")
cppyy.load_library("libgamestream.so")

#%%#

#include <moonlightsdk.h>
cppyy.cppdef("""
#include <stereokit.h>
#include <stereokit_ui.h>
extern "C" {
  #include <libavcodec/avcodec.h>
  #include <libavcodec/jni.h>
  #include <libswscale/swscale.h>
}
""")
# Initialize avcodec with JavaVM pointer
cppyy.cppexec("""
av_jni_set_java_vm(java_vm, NULL);
""")

nsCount = 0

mainHotNs = None
mainHotNsName = None
def reloadMainHot(doSkInit=False):
  global nsCount, mainHotNs, mainHotNsName
  nsCount += 1
  fileContents = open(os.path.expanduser("~/MagicLeap2-Synced/SKTemplate-CMake/src/main_hot.cpp"), "r").read()
  mainHotNsName = f"mainHot{nsCount}"
  cppyy.cppdef(f"""
  namespace {mainHotNsName} {{
    {fileContents}
  }}
  """)
  mainHotNs = getattr(cppyy.gbl, mainHotNsName)
  if (doSkInit):
    mainHotNs.initialize_stereokit()
reloadMainHot(True)
mainHotNs.setup()
mainHotNs.cleanup()

moonNs = None
moonNsName = None
def reloadMoonStream():
  # basically replicating the CMakeLists.txt
  global nsCount, moonNs, moonNsName
  nsCount += 1
  # # stop old stream
  # if moonNs:
  #   cppyy.cppexec(f"""
  #   {moonNsName}::moon_stop_stream();
  #   """)
  fileContents = open(os.path.expanduser("~/MagicLeap2-Synced/moonlight-embedded/src/moonlightsdk.c"), "r").read()
  moonNsName = f"""moonStream{nsCount}"""
  cppyy.cppdef(f"""
  extern "C" {{
    #include <Limelight.h>
    #include <libgamestream/client.h>
    #include <libgamestream/discover.h>
    #include <libgamestream/errors.h>
    #include <libgamestream/http.h>
    #include <libgamestream/mkcert.h>
    #include <libgamestream/sps.h>
    #include <libgamestream/xml.h>

    #include <opus/opus_multistream.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <unistd.h>
    #include <string.h>
    #include <getopt.h>
    #include <pwd.h>
    #include <sys/types.h>
  }}
  // These headers need to be extern C decl. If try to redefine it within the namespace, will get redefition errors. If don't redefine it in subsequent namespaces, its symbols won't be found. Soln is to define outside namespace. These are headers that are included by moonlightsdk.c for two external libs it uses.

  #undef _FFMPEG_H_
  #undef _CONFIG_H_
  // These need to be defined again, since they are pragma once but will only be defined in the first namespace

  namespace {moonNsName} {{
    #include "moonlightsdk.c"
    // include all the cpp files that need to be compiled.
    #include "audio.c"
    #include "config.c"
    #include "connection.c"
    #include "ffmpeg.c"
    #include "video.c"
  }}
  """)
  moonNs = getattr(cppyy.gbl, moonNsName)
  return
  # start new stream, catch errors
  startCode = f"""
  {moonNsName}::moon_init_config("192.168.0.4", "Desktop");
  {moonNsName}::moon_init_server_connection();
  {moonNsName}::moon_start_stream((void*)&{mainHotNsName}::on_moonlight_frame);
  """
  cppyy.cppexec(f"""
  // catch exceptions, including segfaults
  try {{
    {startCode}
  }} catch (...) {{
    printf("Exception in running hot-reloaded moonlightsdk");
  }}
  """)
  # stop stream after 3 seconds
  cppyy.cppexec(f"""
  std::thread t([]() {{
    std::this_thread::sleep_for(std::chrono::seconds(3));
    try {{
      {moonNsName}::moon_stop_stream();
    }} catch (...) {{
      printf("Exception in stopping hot-reloaded moonlightsdk");
    }}
  }});
  t.detach();
  """)

#%%#
import asyncio

reloadMoonStream()

reloadMainHot()
mainHotNs.setup()
# add mainHotNS.update to the jupyter async event loop
# sk init needs to happen on same thread as update, so do this cause it's easier rn
async def mainHotUpdate():
  while True:
    mainHotNs.update()
    await asyncio.sleep(0.00)
task = asyncio.create_task(mainHotUpdate())
task.cancel()


# for i in range(2000):
#   mainHotNs.update()
mainHotNs.cleanup()
#%%#

# ----- MISCELLANEOUS -----

# List MediaCodec codecs
cppyy.cppdef("""
// list codec supported by MediaCodec
void list_codec_5() {
  JNIEnv *env;
  java_vm->GetEnv((void **)&env, JNI_VERSION_1_6);
  jclass media_codec_list = env->FindClass("android/media/MediaCodecList");
  jmethodID get_codec_count = env->GetStaticMethodID(media_codec_list, "getCodecCount", "()I");
  jint codec_count = env->CallStaticIntMethod(media_codec_list, get_codec_count);
  printf("codec_count: %d", codec_count);
  jmethodID get_codec_info_at = env->GetStaticMethodID(media_codec_list, "getCodecInfoAt", "(I)Landroid/media/MediaCodecInfo;");
  for (int i = 0; i < codec_count; i++) {
    jobject codec_info = env->CallStaticObjectMethod(media_codec_list, get_codec_info_at, i);
    jmethodID is_encoder = env->GetMethodID(env->GetObjectClass(codec_info), "isEncoder", "()Z");
    jboolean is_encoder_bool = env->CallBooleanMethod(codec_info, is_encoder);
    if (is_encoder_bool) {
      continue;
    }
    jmethodID get_name = env->GetMethodID(env->GetObjectClass(codec_info), "getName", "()Ljava/lang/String;");
    jstring codec_name = (jstring)env->CallObjectMethod(codec_info, get_name);
    const char *codec_name_cstr = env->GetStringUTFChars(codec_name, NULL);
    printf("codec_name: %s", codec_name_cstr);
    // Check if OMX.mesa.video_decoder.avc is hardware accelerated
    jmethodID get_capabilities_for_type = env->GetMethodID(env->GetObjectClass(codec_info), "getCapabilitiesForType", "(Ljava/lang/String;)Landroid/media/MediaCodecInfo$CodecCapabilities;");
    jthrowable exception = env->ExceptionOccurred();
    if (exception) {
      env->ExceptionDescribe();
      env->ExceptionClear();
      continue;
    } 
    jmethodID is_hardware_accelerated = env->GetMethodID(env->GetObjectClass(codec_info), "isHardwareAccelerated", "()Z");
    exception = env->ExceptionOccurred();
    if (exception) {
      env->ExceptionDescribe();
      env->ExceptionClear();
      continue;
    } 
    jboolean is_hardware_accelerated_bool = env->CallBooleanMethod(codec_info, is_hardware_accelerated);
    if (is_hardware_accelerated_bool) {
      printf("-- %s is hardware accelerated --", codec_name_cstr);
    }
    env->ReleaseStringUTFChars(codec_name, codec_name_cstr);
  }
}
""")
cppyy.gbl.list_codec_5()
