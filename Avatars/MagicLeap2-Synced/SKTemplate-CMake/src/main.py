# Can't Unload/Reload header file if it has #pragma once
# #includes are cached by cling
# If get incrementalExecutor symbol not found on first run, might not get it on subsequent runs, & method will still run, but be terribly broken. (happened on mac). Something todo w/ local symbols?
# Something wrong with llvm::outs either on cppyy or w/ jupyternotebook / vscode. (works fine on command line cling)
# on throw exception, libc++_shared just SIGABRTs
# on JNI exception in sandbox_runner, ssh watch process on termux crashes? maybe?
# linking errors with zmq? not because of initial cppyy ldd stuff. prob because it's loaded by jupyter/pyzmq? And android linker has thing where if load one .so with RTLD_GLOBAL, subsequent loaded .so's (e.g. our libCling.so one) won't be able to find it's symbols. Even if libCling.so dlopens libzmq.so, it's dlsym still fails. 
# - note: use cppyy.gbl.gInterpreter.Load("libpath") to load .so's. cppyy.load_library calls gSystem.Load, which calls gInterpreter.Load, which calls llvm::sys::... , will miss errors otherwise.
# - trying patchelf --add-needed /data/data/com.termux/files/usr/lib/libzmq.so libCling.so -- it works! dlopening with libzmq that's renamed also works.
# - seems like only reliable way of dlsym on android is to either dlsym with handle from dlopen, or add-needed the lib to the .so that's doing the dlsyming. Because here's what happened: python imported pyzmq which in it's _backend.so has so dependency libzmq. Then, python imported/dlopened libcling.so. libcling.so tried to access libzmq symbols with both dlsym(RTLD_DEFAULT, ...) and dlsym(dlopen(NULL, ...), ...) which didn't work. Also tried having libcling.so dlopen libzmq.so, and the above two methods still didn't work.

#%%#

import cppyy
import os
import asyncio
import json
import datetime
import time

# import jupyter display
from IPython.display import display, Javascript, HTML
from html import escape

# import clang.cindex
import shlex
import ctypes
cppyy.add_include_path("/opt/homebrew/opt/llvm@14/include")
cppyy.gbl.gInterpreter.Load("/opt/homebrew/Cellar/llvm@14/14.0.6/lib/libLLVM.dylib")
cppyy.gbl.gInterpreter.Load("/opt/homebrew/Cellar/llvm@14/14.0.6/lib/libclang-cpp.dylib")
cppyy.gbl.gInterpreter.Load("/opt/homebrew/Cellar/llvm@14/14.0.6/lib/libclang.dylib")
cppyy.cppdef("""
  #include "clang-c/Index.h"
  #include "clang-c/Rewrite.h"
""")

# cppyy.cppdef("""
# #include <stdio.h>
# #include <stdlib.h>
# #include <android/log.h>
# // fake printf using android log
# #define printf(...) __android_log_print(ANDROID_LOG_INFO, "Sunflower cppyy", __VA_ARGS__)
# """)

globalNsCount = 1
def defineInNewNs(contents):
  global globalNsCount
  globalNsCount += 1
  nsName = f"sfGlobalNs{globalNsCount}"
  cppyy.cppdef(f"""
  namespace {nsName} {{
    {contents}
  }}
  """)
  return (nsName, getattr(cppyy.gbl, nsName))

def includeFile(dir, path, runtimeNsName, transform=False):
  clangNs = cppyy.gbl
  fullPath = os.path.join(dir, path)
  # fullPath = "/Users/cameronfranz/Documents/Projects/Sunflower/SunflowerOS/Repo/Avatars/MagicLeap2-Synced/SKTemplate-CMake/src/SunflowerEditorRuntime.cpp" 
  fileContents = open(fullPath, "r").read()
  # Basically want output of cppyy.gbl.gInterpreter.ProcessLine(".I")
  clangArgs = shlex.split(cppyy.gbl.gInterpreter.GetIncludePath()) + ["-resource-dir", cppyy.gbl.CppyyLegacy.GetROOT().GetEtcDir().Data() + "cling/lib/clang/9.0.1"]  
  index2 = clangNs.clang_createIndex(0, 0)
  tu2 = clangNs.clang_parseTranslationUnit(index2, fullPath.encode("utf-8"), clangArgs, len(clangArgs), cppyy.nullptr, 0, 0)

  def CXtoCStr(cxstr):
    string = clangNs.clang_getCString(cxstr)
    clangNs.clang_disposeString(cxstr)
    return string

  for i in range(clangNs.clang_getNumDiagnostics(tu2)):
    diag = clangNs.clang_getDiagnostic(tu2, i)
    diagStr = CXtoCStr(clangNs.clang_formatDiagnostic(diag, clangNs.clang_defaultDiagnosticDisplayOptions()))
    diagSeverity = clangNs.clang_getDiagnosticSeverity(diag)
    # print("clang diagnostic: " + diagStr)
    if (diagSeverity > clangNs.CXDiagnostic_Warning):
      print("clang diagnostic: " + diagStr)
      raise Exception("Clang error: ")
    clangNs.clang_disposeDiagnostic(diag)
  
  _, tmp = defineInNewNs("""
    CXCursor* clang_makeCursorCopy(CXCursor* cursor) {
      CXCursor* newCursor = new CXCursor();
      *newCursor = *cursor;
      return newCursor;
    }
  """)
  def getCursorChildren(cursor):
    children = []
    def visit(c, p, l):
      # clang_visitChildren reuses same CXCursor
      cCopy = tmp.clang_makeCursorCopy(c)
      children.append(cCopy)
      return clangNs.CXChildVisit_Continue
    clangNs.clang_visitChildren(cursor, visit, cppyy.nullptr)
    return children
  getCursorChildren(clangNs.clang_getTranslationUnitCursor(tu2))

  cxrewriter = clangNs.clang_CXRewriter_create(tu2)
  def visitor(cursor, depth):
    location = clangNs.clang_getCursorLocation(cursor)
    isFromMainFile = clangNs.clang_Location_isFromMainFile(location)
    if not isFromMainFile:
      return 
    
    clangNs.clang_Location_isFromMainFile
    kind = clangNs.clang_getCursorKind(cursor)
    spelling = CXtoCStr(clangNs.clang_getCursorSpelling(cursor))
    kind = clangNs.clang_getCursorKind(cursor)
    kindReadable = CXtoCStr(clangNs.clang_getCursorKindSpelling(kind))

    line = ctypes.c_uint32()
    column = ctypes.c_uint32()
    clangNs.clang_getExpansionLocation(location, ctypes.c_void_p(), line, column, cppyy.nullptr)
    extent = clangNs.clang_getCursorExtent(cursor)

    # check if it's a printf
    if kind == clangNs.CXCursor_CallExpr:
      if spelling == "printf":
        argStrings = []
        # e.g. printf("hello %d", 5) -> ["printf", "\"hello %d\"", "5"]
        for argCursor in getCursorChildren(cursor):
          argExtent = clangNs.clang_getCursorExtent(argCursor)
          start = clangNs.clang_getRangeStart(argExtent)
          end = clangNs.clang_getRangeEnd(argExtent)
          startIdx = ctypes.c_uint32()
          endIdx = ctypes.c_uint32()
          _file = ctypes.c_void_p()
          clangNs.clang_getExpansionLocation(start, _file, cppyy.nullptr, cppyy.nullptr, startIdx)
          clangNs.clang_getExpansionLocation(end, _file, cppyy.nullptr, cppyy.nullptr, endIdx)
          extentStr = fileContents[startIdx.value:endIdx.value]
          argStrings.append(extentStr) 
          # print("extentStr: " + extentStr)
    
        vaArgs = ", ".join(argStrings[1:])
        truncPath = "/".join(fullPath.split("/")[-3:])
        newCode = f"""
          {{
            int sf_bufferSize = snprintf(NULL, 0, {vaArgs}); 
            char *sf_buffer = (char*)malloc(sf_bufferSize + 1); 
            snprintf(sf_buffer, sf_bufferSize + 1, {vaArgs}); 
            {runtimeNsName}::msgserver_inthread_sendlog("{truncPath}", {line.value-1}, {column.value-1}, sf_buffer); 
            free(sf_buffer); 
          }}
        """
        clangNs.clang_CXRewriter_replaceText(cxrewriter, extent, newCode)
    for child in getCursorChildren(cursor):
      visitor(child, depth+1)

  if transform:
    for child in getCursorChildren(clangNs.clang_getTranslationUnitCursor(tu2)):
      visitor(child, 0)

  cppyy.cppdef("""
    #include "clang/Basic/SourceManager.h"
    #include "clang/Frontend/ASTUnit.h"
    #include "clang/Rewrite/Core/Rewriter.h"
  """)
  _, tmp = defineInNewNs("""
    std::string getRewriteBuffer(CXRewriter rew) {
      clang::Rewriter &rewriter = *(clang::Rewriter*)rew;
      // get the rewritten buffer
      std::string buffer;
      llvm::raw_string_ostream os(buffer);
      rewriter.getEditBuffer(rewriter.getSourceMgr().getMainFileID()).write(os);
      return os.str();
    }
  """)
  transformedFileContents = tmp.getRewriteBuffer(cxrewriter)
  # create jupyter scrollable output
  display(HTML(f"""
    <style>
      .sf-code-output {{
        max-height: 200px;
        overflow-y: scroll;
      }}
    </style>
    <div class="sf-code-output">
      <pre>{escape(transformedFileContents)}</pre>
    </div>
  """))
  # print(transformedFileContents)

  return transformedFileContents

# editorRuntimeDir = os.path.expanduser("~/MagicLeap2-Synced/SKTemplate-CMake/src")
editorRuntimeDir = "/Users/cameronfranz/Documents/Projects/Sunflower/SunflowerOS/Repo/Avatars/MagicLeap2-Synced/SKTemplate-CMake/src"
# cppyy.gbl.gInterpreter.Load("/data/data/com.termux/files/usr/lib/libzmq_testrename.so", True)
cppyy.load_library("libzmq.so")
cppyy.cppdef("""
#include <zmq.h>
#include <math.h>
""")

# TODO: segfaulting in zmq sometimes
# __android_log_print calls seem to take about 5us each
editorRuntimeNsName, editorRuntime = defineInNewNs(f"""
  {includeFile(editorRuntimeDir, "SunflowerEditorRuntime.cpp", runtimeNsName=None, transform=False)}
""")
editorRuntime.msgserver_init()
for i in range(1000):
  editorRuntime.msgserver_inthread_sendlog("test/main_hot.cpp", 10, 4, "hello world"+datetime.datetime.now().strftime("%H:%M:%S.%f"))
  # time.sleep(0.001)
editorRuntime.msgserver_close()

_, runtimeTest = defineInNewNs(f"""
  {includeFile(editorRuntimeDir, "SunflowerEditorTesting.cpp", runtimeNsName=editorRuntimeNsName, transform=True)}
""")
runtimeTest.doStuff()
for i in range(100):
  runtimeTest.doStuff2()


# import cppyy.ll
# cppyy.ll.set_signals_as_exception(True) # performance impact apparently. catches some segfaults and JNI faults, but seems state is messed up after catch those and have to restart kernel anyways. c++ exceptions still broken.

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
cppyy.add_include_path(os.path.expanduser("~/libyuv/include"))
cppyy.add_library_path(os.path.expanduser("~/libyuv/"))
# cppyy.add_library_path(os.path.expanduser("~/MagicLeap2-Synced/moonlight-embedded/"))
# cppyy.load_library("libmoonlight.so")
cppyy.load_library("libopus.so")
cppyy.load_library("libyuv.so")
cppyy.load_library("libgamestream.so")

#%%#

cppyy.cppdef("""
#include <stereokit.h>
#include <stereokit_ui.h>
extern "C" {
}
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
  srcPath = os.path.expanduser("~/MagicLeap2-Synced/moonlight-embedded/src/")
  includeFile = lambda name: open(os.path.join(srcPath, name), "r").read()
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
    #include <libyuv.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <unistd.h>
    #include <string.h>
    #include <getopt.h>
    #include <pwd.h>
    #include <sys/types.h>
  }}
  // These headers need to be extern C decl. If try to redefine it within the namespace, will get redefition errors. If don't redefine it in subsequent namespaces, its symbols won't be found. Soln is to define outside namespace. These are headers that are included by moonlightsdk.c for two external libs it uses.

  #undef _CONFIG_H_
  // These need to be defined again, since they are pragma once but will only be defined in the first namespace

  namespace {moonNsName} {{
    // include all the cpp files that need to be compiled.
    // if use #include "...", cling will cache the file contents
    {includeFile("moonlightsdk.c")}
    {includeFile("audio.c")}
    {includeFile("config.c")}
    {includeFile("connection.c")}
    {includeFile("video.c")}
  }}
  """)
  moonNs = getattr(cppyy.gbl, moonNsName)
  # start new stream, catch errors
  cppyy.cppdef(f"""
  namespace {moonNsName} {{
    void testStart() {{
      try {{
        {moonNsName}::moon_init_config("192.168.0.4", "Desktop");
        {moonNsName}::moon_init_server_connection();
        {moonNsName}::moon_start_stream((void*)&{mainHotNsName}::on_moonlight_frame);
      }} catch(...) {{
        printf("Error testing start, caught exception");
      }}
    }}
    void testStop() {{
      try {{
        {moonNsName}::moon_stop_stream();
      }} catch(...) {{
        printf("Error testing stop, caught exception");
      }}
    }}
  }}
  """)

#%%#

reloadMainHot()
mainHotNs.setup()

reloadMoonStream()
moonNs.testStart()
# add mainHotNS.update to the jupyter async event loopp
# sk init needs to happen on same thread as update, so do this cause it's easier rn
async def mainHotUpdate():
  while True:
    mainHotNs.update()
    await asyncio.sleep(0.00)
task = asyncio.create_task(mainHotUpdate())
await asyncio.sleep(30)
task.cancel()
moonNs.testStop()

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

import cppyy
nsCount = 0
cppyy.cppdef(f"""
#include <jni.h> 
bool checkJNIError(JNIEnv *env) {{
  if (env->ExceptionCheck()) {{
    env->ExceptionDescribe();
    env->ExceptionClear();
    return true;
  }}
  return false;
}}
""")

nsCount += 1
cppyy.cppdef(f"""
#include <jni.h>
void codec_info_{nsCount}() {{
  JNIEnv *env;
  java_vm->GetEnv((void **)&env, JNI_VERSION_1_6);
  // get OMX.mesa.video_decoder.avc 
  jclass media_codec = env->FindClass("android/media/MediaCodec");
  jmethodID create_by_codec_name = env->GetStaticMethodID(media_codec, "createByCodecName", "(Ljava/lang/String;)Landroid/media/MediaCodec;");
  jstring codec_name = env->NewStringUTF("OMX.mesa.video_decoder.avc");
  jobject codec = env->CallStaticObjectMethod(media_codec, create_by_codec_name, codec_name);
  // get codec info
  jmethodID get_codec_info = env->GetMethodID(media_codec, "getCodecInfo", "()Landroid/media/MediaCodecInfo;");
  jobject codec_info = env->CallObjectMethod(codec, get_codec_info);
  // get codec capabilities
  jmethodID get_capabilities_for_type = env->GetMethodID(env->GetObjectClass(codec_info), "getCapabilitiesForType", "(Ljava/lang/String;)Landroid/media/MediaCodecInfo$CodecCapabilities;");
  jstring mime_type = env->NewStringUTF("video/avc");
  jobject codec_capabilities = env->CallObjectMethod(codec_info, get_capabilities_for_type, mime_type);
  // check if supports fused idr using isFeatureSupported(CodecCapabilities.FEATURE_LowLatency)
  jmethodID is_feature_supported = env->GetMethodID(env->GetObjectClass(codec_capabilities), "isFeatureSupported", "(Ljava/lang/String;)Z");
  jstring feature_low_latency = env->NewStringUTF("low-latency");
  jboolean is_feature_supported_bool = env->CallBooleanMethod(codec_capabilities, is_feature_supported, feature_low_latency);
  if (is_feature_supported_bool) {{
    printf("OMX.mesa.video_decoder.avc supports FEATURE_LowLatency\\n");
  }} else {{
    printf("OMX.mesa.video_decoder.avc does not support FEATURE_LowLatency\\n");
  }}
  // check if supports adaptive playback using isFeatureSupported(CodecCapabilities.FEATURE_AdaptivePlayback)
  jstring feature_adaptive_playback = env->NewStringUTF("adaptive-playback");
  is_feature_supported_bool = env->CallBooleanMethod(codec_capabilities, is_feature_supported, feature_adaptive_playback);
  if (is_feature_supported_bool) {{
    printf("OMX.mesa.video_decoder.avc supports FEATURE_AdaptivePlayback\\n");
  }} else {{
    printf("OMX.mesa.video_decoder.avc does not support FEATURE_AdaptivePlayback\\n");
  }}
  // get supported vendor parameters getSupportedVendorParameters
  jmethodID get_supported_vendor_parameters = env->GetMethodID(media_codec, "getSupportedVendorParameters", "()Ljava/util/List<Ljava/lang/String;>;");
  if (checkJNIError(env)) return;
  jobject supported_vendor_parameters = env->CallObjectMethod(codec, get_supported_vendor_parameters);
  if (checkJNIError(env)) return;
  // list supported vendor parameters
  jmethodID get_size = env->GetMethodID(env->GetObjectClass(supported_vendor_parameters), "size", "()I");
  if (checkJNIError(env)) return;
  jint size = env->CallIntMethod(supported_vendor_parameters, get_size);
  if (checkJNIError(env)) return;
  jmethodID get = env->GetMethodID(env->GetObjectClass(supported_vendor_parameters), "get", "(I)Ljava/lang/Object;");
  if (checkJNIError(env)) return;
  for (int i = 0; i < size; i++) {{
    jstring vendor_parameter = (jstring)env->CallObjectMethod(supported_vendor_parameters, get, i);
    if (checkJNIError(env)) return;
    const char *vendor_parameter_cstr = env->GetStringUTFChars(vendor_parameter, NULL);
    if (checkJNIError(env)) return;
    printf("OMX.mesa.video_decoder.avc supports vendor parameter %s\\n", vendor_parameter_cstr);
    env->ReleaseStringUTFChars(vendor_parameter, vendor_parameter_cstr);
  }}
}}
void codec_info_wrapped_{nsCount}() {{
  try {{
    // catch jni exceptions
    codec_info_{nsCount}();
  }} catch (...) {{
    printf("codec_info_wrapped_{nsCount} failed");
  }}
}}
""")
method = getattr(cppyy.gbl, f"codec_info_wrapped_{nsCount}")
method()


# Test throwing exception
# cppyy.cppexec("""
# try {
#   throw std::runtime_error("test");
# } catch(...) {
#   printf("caught exception");
# }
# """)
