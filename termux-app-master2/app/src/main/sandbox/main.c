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

JavaVM *vm = NULL;

jint JNI_OnLoad(JavaVM *vmIn, void *reserved) {
    vm = vmIn;
    LOGI("JNI_OnLoad in SandboxRunner so %p", vm);
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL
Java_com_termux_SandboxRunner_loadNativeCode(JNIEnv *env, jobject thiz, jstring path, jstring funcname) {
  // split path string into libpath and libargs with space
  LOGE("Using strtok");
  char *pathStr = (char *) (*env)->GetStringUTFChars(env, path, NULL);
  char *libPath = strtok(pathStr, " ");
  char *libArgs = strtok(NULL, "");
  // create fake argc and argv by splitting libargs with space
  int argc = 0;
  char *argv[100];
  argv[argc++] = libPath;
  char *arg = strtok(libArgs, " ");
  while (arg != NULL) {
    argv[argc++] = arg;
    arg = strtok(NULL, " ");
  }

  void *handle = dlopen(libPath, RTLD_LAZY);
    if (handle) {
      // Call JNI_OnLoad
      void (*JNI_OnLoad)(JavaVM *, void *) = (void (*)(JavaVM *, void *)) dlsym(handle, "JNI_OnLoad");
      if (JNI_OnLoad) {
        JNI_OnLoad(vm, NULL);
      } else {
        LOGE("JNI_OnLoad not found");
      }

      // Call JNI_SetActivity (not standard)
      void (*JNI_SetActivity)(jobject) = (void (*)(jobject)) dlsym(handle, "JNI_SetActivity");
      if (JNI_SetActivity) {
        jobject activity = (*env)->NewGlobalRef(env, thiz);
        JNI_SetActivity(activity); //on ML2 XrInstanceCreateInfoAndroidKHR doesn't actually care what this is (can be bogus)??
      } else {
        LOGE("JNI_SetActivity not found");
      }

      void (*mainFunc)(int, char**) = dlsym(handle, "main");
      if (mainFunc) {
        // mainFunc(0, NULL);
        mainFunc(argc, argv);
      } else {
        LOGI("SandboxRunner dlserver main not found");
      }
      dlclose(handle);
    } else {
      LOGI("SandboxRunner dlserver dlopen failed because %s", dlerror());
    }
}
