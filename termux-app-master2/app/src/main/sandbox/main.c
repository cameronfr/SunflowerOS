#include <jni.h>
#include <unistd.h>
#include <errno.h>
#include <dirent.h>
#include <dlfcn.h>
#include <link.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <android/log.h>

#define LOG_TAG "SunflowerSandboxRunner"

#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

JNIEXPORT void JNICALL
Java_com_termux_SandboxRunner_loadNativeCode(JNIEnv *env, jobject thiz, jstring path, jstring funcname) {
  char *libname = (char *) (*env)->GetStringUTFChars(env, path, NULL);
  void *handle = dlopen(libname, RTLD_LAZY); 
  if (handle) {
    void (*main)() = dlsym(handle, "main");
    if (main) {
      // TODO: call JNI_OnLoad (?)
      main();
    } else {
      LOGE("main not found");
    }
    dlclose(handle);
  } else {
    LOGI("dlopen failed because %s", dlerror());
  }
}
