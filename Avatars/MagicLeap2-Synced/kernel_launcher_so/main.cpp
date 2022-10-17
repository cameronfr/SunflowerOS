// Packages python code as an .so

#include "Python.h"

#include <stdio.h>
#include <stdlib.h>
#include <android/log.h>
#include <jni.h>

// Redirects stdout, stderr to android log. 
// https://codelab.wordpress.com/2014/11/03/how-to-use-standard-output-streams-for-logging-in-android-apps/
static int pfd[2];
static pthread_t thr;
static const char *tag = "Sunflower kernel_launcher_so";
static void *thread_func(void*)
{
    ssize_t rdsz;
    char buf[512];
    while((rdsz = read(pfd[0], buf, sizeof buf - 1)) > 0) {
        if(buf[rdsz - 1] == '\n') --rdsz;
        buf[rdsz] = 0;  /* add null-terminator */
        __android_log_write(ANDROID_LOG_DEBUG, tag, buf);
    }
    return 0;
}
int start_logger(const char *app_name)
{
    tag = app_name;

    /* make stdout line-buffered and stderr unbuffered */
    setvbuf(stdout, 0, _IOLBF, 0);
    setvbuf(stderr, 0, _IONBF, 0);

    /* create the pipe and redirect stdout and stderr */
    pipe(pfd);
    dup2(pfd[1], 1);
    dup2(pfd[1], 2);

    /* spawn the logging thread */
    if(pthread_create(&thr, 0, thread_func, 0) == -1)
        return -1;
    pthread_detach(thr);
    return 0;
}

// Get Java vm pointer
void *java_vm;
extern "C" jint JNI_OnLoad(JavaVM *vm, void *reserved) {
  java_vm = vm;
  return JNI_VERSION_1_6;
} 

int main(int argc, char *argv[]) {
  start_logger("Sunflower kernel_launcher_so");

  printf("Sunflower kernel_launcher_so got %d args\n", argc);
  for (int i = 0; i < argc; i++) {
    printf("Arg %d: %s\n", i, argv[i]);
  }

  // If launch from forked dalvik process, we won't have PATH set.
  // PATH used for sys.executable in python
  // cppyy needs stuff added in LD_LIBRARY_PATH
  setenv("PATH", "/data/data/com.termux/files/usr/bin", 1);
  setenv("LD_LIBRARY_PATH", "/data/data/com.termux/files/usr/lib/python3.10/site-packages/cppyy_backend/lib", 1);
  // set Java vm pointer as env var that we can access in python / cppyy

  // have to convert it to null terminated string first
  char buf[100];
  sprintf(buf, "%p", java_vm);
  setenv("JAVA_VM_PTR", buf, 1);

  Py_Initialize();
  // Fake argc, argv for python (since launch_new_instance expects sys.argv to have "-f kernel-<id>.json")
  // Make sure to convert to wchar_t
  wchar_t *wargv[argc];
  for (int i = 0; i < argc; i++) {
    wargv[i] = Py_DecodeLocale(argv[i], NULL);
  }
  PySys_SetArgv(argc, wargv);

  char *pythonCode = 
    "import sys\n"
    "print('Sys.executable: ' + sys.executable)\n"
    "print('Sys.argv: ' + str(sys.argv))\n"
    "print('Sys.path: ' + str(sys.path))\n"
    "#sys.argv = \"python -f kernel-12939842.json\".split(\" \")\n"
    "from ipykernel import kernelapp as app\n"
    "app.launch_new_instance()\n";
  PyRun_SimpleString(pythonCode);
  Py_Finalize();
  return 0;
} 