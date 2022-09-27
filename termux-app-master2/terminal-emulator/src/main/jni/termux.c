#include <dirent.h>
#include <fcntl.h>
#include <jni.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <termios.h>
#include <unistd.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <dlfcn.h>

#include <android/log.h>
#include <android/native_activity.h>
#include <android/native_window_jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#define TAG "JNIDEBUG"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,     TAG, __VA_ARGS__)

#define TERMUX_UNUSED(x) x __attribute__((__unused__))
#ifdef __APPLE__
# define LACKS_PTSNAME_R
#endif
JavaVM *android_java_vm;

/////////////////////////////

jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    android_java_vm = vm;
    return JNI_VERSION_1_6;
}
jint JNI_OnLoad_L(JavaVM* vm, void* reserved) {
    return JNI_OnLoad(vm, reserved);
}

////////////////////////////////////

void launchRunnerActivity(char *libName) {
    printf("%s", libName);
  JNIEnv *env;
  int result = (*android_java_vm)->GetEnv(android_java_vm, (void **)&env, JNI_VERSION_1_6);
  if (result != JNI_OK) {
      LOGI("JNIDebug Couldn't get the Java Environment from the VM, this needs to be called from the main thread.");
  }
  jclass activityThread = (*env)->FindClass(env,"android/app/ActivityThread");
  jmethodID currentActivityThread = (*env)->GetStaticMethodID(env,activityThread, "currentActivityThread", "()Landroid/app/ActivityThread;");
  jobject at = (*env)->CallStaticObjectMethod(env,activityThread, currentActivityThread);
  jmethodID getApplication = (*env)->GetMethodID(env,activityThread, "getApplication", "()Landroid/app/Application;");
  jobject context_inst = (*env)->CallObjectMethod(env,at, getApplication);
  jobject context = context_inst;

  jclass native_context = (*env)->GetObjectClass(env, context);
  jclass intentClass = (*env)->FindClass(env, "android/content/Intent");
  jclass actionString = (*env)->FindClass(env, "com/termux/SandboxRunner");
  jmethodID newIntent = (*env)->GetMethodID(env, intentClass, "<init>", "(Landroid/content/Context;Ljava/lang/Class;)V");
  jobject intent = (*env)->NewObject(env, intentClass,newIntent,context,actionString);

  // Add extra to intent
  jmethodID putExtra = (*env)->GetMethodID(env, intentClass, "putExtra", "(Ljava/lang/String;Ljava/lang/String;)Landroid/content/Intent;");
  jstring extraName = (*env)->NewStringUTF(env, "libName");
  jstring extraValue = (*env)->NewStringUTF(env, libName);
  (*env)->CallObjectMethod(env, intent, putExtra, extraName, extraValue);

  // Start activity
  jmethodID methodFlag = (*env)->GetMethodID(env, intentClass, "setFlags", "(I)Landroid/content/Intent;");
  jobject intentActivity = (*env)->CallObjectMethod(env, intent, methodFlag, 268435456 );
  jmethodID startActivityMethodId = (*env)->GetMethodID(env, native_context, "startActivity", "(Landroid/content/Intent;)V");
  (*env)->CallVoidMethod(env, context, startActivityMethodId, intentActivity);
}

void launchDLOPENSocketServer() {
    char *filename = "/data/data/com.termux/files/dlopensocketserver.sock";
    struct sockaddr_un name;
    int sock;
    size_t size;

    sock = socket(PF_LOCAL, SOCK_DGRAM, 0);
    if (sock < 0) {
        LOGI("SandboxRunner socket create failed because %s", strerror(errno));
        perror ("socket failed to create ");
        exit (EXIT_FAILURE);
    }

    /* Bind a name to the socket. */
    name.sun_family = AF_LOCAL;
    strncpy (name.sun_path, filename, sizeof (name.sun_path));
    name.sun_path[sizeof (name.sun_path) - 1] = '\0';

    size = (offsetof (struct sockaddr_un, sun_path)
            + strlen (name.sun_path));

    unlink (filename);
    if (bind(sock, (struct sockaddr *) &name, size) < 0) {
        LOGI("SandboxRunner socketserver bind failed because %s", strerror(errno));
        perror ("socket bind failed");
        exit (EXIT_FAILURE);
    }

    LOGI("SandboxRunner socket server starting");

    // Listen on socket, and print what is received
    char buf[1024];
    while (1) {
        int len = recv(sock, buf, sizeof(buf), 0);
        if (len < 0) {
            perror("recv");
            exit(1);
        }
        buf[len] = '\0';
        LOGI("SandboxRunner Socket Received: %s", buf);
        char *libname = buf;

        // Launch the SandboxRunner android activity with the library name
        launchRunnerActivity(libname);
    }
}

void launchWithDLOpen(char *libname) {
    void *handle = dlopen(libname, RTLD_LAZY);
    if (handle) {
        void (*main)() = dlsym(handle, "main");
        if (main) {
            // TODO: call JNI_OnLoad (?)
            main();
        } else {
            LOGI("main not found");
        }
        dlclose(handle);
    } else {
        LOGI("dlopen failed because %s", dlerror());
    }
}

static int throw_runtime_exception(JNIEnv* env, char const* message)
{
    jclass exClass = (*env)->FindClass(env, "java/lang/RuntimeException");
    (*env)->ThrowNew(env, exClass, message);
    return -1;
}

static int create_subprocess(JNIEnv* env,
        char const* cmd,
        char const* cwd,
        char* const argv[],
        char** envp,
        int* pProcessId,
        jint rows,
        jint columns)
{
    int ptm = open("/dev/ptmx", O_RDWR | O_CLOEXEC);
    if (ptm < 0) return throw_runtime_exception(env, "Cannot open /dev/ptmx");

#ifdef LACKS_PTSNAME_R
    char* devname;
#else
    char devname[64];
#endif
    if (grantpt(ptm) || unlockpt(ptm) ||
#ifdef LACKS_PTSNAME_R
            (devname = ptsname(ptm)) == NULL
#else
            ptsname_r(ptm, devname, sizeof(devname))
#endif
       ) {
        return throw_runtime_exception(env, "Cannot grantpt()/unlockpt()/ptsname_r() on /dev/ptmx");
    }

    // Enable UTF-8 mode and disable flow control to prevent Ctrl+S from locking up the display.
    struct termios tios;
    tcgetattr(ptm, &tios);
    tios.c_iflag |= IUTF8;
    tios.c_iflag &= ~(IXON | IXOFF);
    tcsetattr(ptm, TCSANOW, &tios);

    /** Set initial winsize. */
    struct winsize sz = { .ws_row = (unsigned short) rows, .ws_col = (unsigned short) columns };
    ioctl(ptm, TIOCSWINSZ, &sz);

    pid_t pid = fork();
    if (pid < 0) {
        return throw_runtime_exception(env, "Fork failed");
    } else if (pid > 0) {
        *pProcessId = (int) pid;
        //launchRunnerActivity("/data/data/com.termux/files/home/MagicLeap2-Synced/feature_testing/libFeatureTestLib.so");
        //launchWithDLOpen("/data/data/com.termux/files/home/MagicLeap2-Synced/feature_testing/libFeatureTestLib.so");

        launchDLOPENSocketServer();
        // IDK where this ends up, since socket server listens forever, but guess it's not important
        return ptm;
    } else {
        // Clear signals which the Android java process may have blocked:
        sigset_t signals_to_unblock;
        sigfillset(&signals_to_unblock);
        sigprocmask(SIG_UNBLOCK, &signals_to_unblock, 0);

        close(ptm);
        setsid();

        int pts = open(devname, O_RDWR);
        if (pts < 0) exit(-1);

        dup2(pts, 0);
        dup2(pts, 1);
        dup2(pts, 2);

        DIR* self_dir = opendir("/proc/self/fd");
        if (self_dir != NULL) {
            int self_dir_fd = dirfd(self_dir);
            struct dirent* entry;
            LOGI("JNIDEBUG fork-child entering loop");
            while ((entry = readdir(self_dir)) != NULL) {
                int fd = atoi(entry->d_name);
                LOGI("JNIDEBUG fork-child fd is %d, fmame is %s, self_dir_fd is %d", fd, entry->d_name, self_dir_fd);
                if (fd > 2 && fd != self_dir_fd) close(fd); //for some reason this causes logging to stop
            }
            LOGI("JNIDEBUG fork-child exiting loop");
            closedir(self_dir);
        }
        LOGI("JNIDEBUG at fork-child");
//        JavaVM *failVM = (void*)0x783d82b25b56;
//        JNIEnv *jniEnv = 0; int result = (*failVM)->GetEnv(failVM, (void **)&jniEnv, JNI_VERSION_1_6);
         JNIEnv *jniEnv = 0; int result = (*android_java_vm)->GetEnv(android_java_vm, (void **)&jniEnv, JNI_VERSION_1_6);
        LOGI("JNIDEBUG fork-child jniEnv pointer res %d, val is %p", result, jniEnv);

        clearenv();
        if (envp) for (; *envp; ++envp) putenv(*envp);

        if (chdir(cwd) != 0) {
            char* error_message;
            // No need to free asprintf()-allocated memory since doing execvp() or exit() below.
            if (asprintf(&error_message, "chdir(\"%s\")", cwd) == -1) error_message = "chdir()";
            perror(error_message);
            fflush(stderr);
        }
        execvp(cmd, argv);
        // Show terminal output about failing exec() call:
        char* error_message;
        if (asprintf(&error_message, "exec(\"%s\")", cmd) == -1) error_message = "exec()";
        perror(error_message);
        _exit(1);
    }
}

JNIEXPORT jint JNICALL Java_com_termux_terminal_JNI_createSubprocess(
        JNIEnv* env,
        jclass TERMUX_UNUSED(clazz),
        jstring cmd,
        jstring cwd,
        jobjectArray args,
        jobjectArray envVars,
        jintArray processIdArray,
        jint rows,
        jint columns)
{
    jsize size = args ? (*env)->GetArrayLength(env, args) : 0;
    char** argv = NULL;
    if (size > 0) {
        argv = (char**) malloc((size + 1) * sizeof(char*));
        if (!argv) return throw_runtime_exception(env, "Couldn't allocate argv array");
        for (int i = 0; i < size; ++i) {
            jstring arg_java_string = (jstring) (*env)->GetObjectArrayElement(env, args, i);
            char const* arg_utf8 = (*env)->GetStringUTFChars(env, arg_java_string, NULL);
            if (!arg_utf8) return throw_runtime_exception(env, "GetStringUTFChars() failed for argv");
            argv[i] = strdup(arg_utf8);
            (*env)->ReleaseStringUTFChars(env, arg_java_string, arg_utf8);
        }
        argv[size] = NULL;
    }

    size = envVars ? (*env)->GetArrayLength(env, envVars) : 0;
    char** envp = NULL;
    if (size > 0) {
        envp = (char**) malloc((size + 1) * sizeof(char *));
        if (!envp) return throw_runtime_exception(env, "malloc() for envp array failed");
        for (int i = 0; i < size; ++i) {
            jstring env_java_string = (jstring) (*env)->GetObjectArrayElement(env, envVars, i);
            char const* env_utf8 = (*env)->GetStringUTFChars(env, env_java_string, 0);
            LOGI("JNIDEBUG env_string %d: %s", i, env_utf8);
            if (!env_utf8) return throw_runtime_exception(env, "GetStringUTFChars() failed for env");
            envp[i] = strdup(env_utf8);
            (*env)->ReleaseStringUTFChars(env, env_java_string, env_utf8);
        }
        envp[size] = NULL;
    }

    int procId = 0;
    char const* cmd_cwd = (*env)->GetStringUTFChars(env, cwd, NULL);
    char const* cmd_utf8 = (*env)->GetStringUTFChars(env, cmd, NULL);
    int ptm = create_subprocess(env, cmd_utf8, cmd_cwd, argv, envp, &procId, rows, columns);
    (*env)->ReleaseStringUTFChars(env, cmd, cmd_utf8);
    (*env)->ReleaseStringUTFChars(env, cmd, cmd_cwd);

    if (argv) {
        for (char** tmp = argv; *tmp; ++tmp) free(*tmp);
        free(argv);
    }
    if (envp) {
        for (char** tmp = envp; *tmp; ++tmp) free(*tmp);
        free(envp);
    }

    int* pProcId = (int*) (*env)->GetPrimitiveArrayCritical(env, processIdArray, NULL);
    if (!pProcId) return throw_runtime_exception(env, "JNI call GetPrimitiveArrayCritical(processIdArray, &isCopy) failed");

    *pProcId = procId;
    (*env)->ReleasePrimitiveArrayCritical(env, processIdArray, pProcId, 0);

    return ptm;
}

JNIEXPORT void JNICALL Java_com_termux_terminal_JNI_setPtyWindowSize(JNIEnv* TERMUX_UNUSED(env), jclass TERMUX_UNUSED(clazz), jint fd, jint rows, jint cols)
{
    struct winsize sz = { .ws_row = (unsigned short) rows, .ws_col = (unsigned short) cols };
    ioctl(fd, TIOCSWINSZ, &sz);
}

JNIEXPORT void JNICALL Java_com_termux_terminal_JNI_setPtyUTF8Mode(JNIEnv* TERMUX_UNUSED(env), jclass TERMUX_UNUSED(clazz), jint fd)
{
    struct termios tios;
    tcgetattr(fd, &tios);
    if ((tios.c_iflag & IUTF8) == 0) {
        tios.c_iflag |= IUTF8;
        tcsetattr(fd, TCSANOW, &tios);
    }
}

JNIEXPORT jint JNICALL Java_com_termux_terminal_JNI_waitFor(JNIEnv* TERMUX_UNUSED(env), jclass TERMUX_UNUSED(clazz), jint pid)
{
    int status;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        return -WTERMSIG(status);
    } else {
        // Should never happen - waitpid(2) says "One of the first three macros will evaluate to a non-zero (true) value".
        return 0;
    }
}

JNIEXPORT void JNICALL Java_com_termux_terminal_JNI_close(JNIEnv* TERMUX_UNUSED(env), jclass TERMUX_UNUSED(clazz), jint fileDescriptor)
{
    close(fileDescriptor);
}
