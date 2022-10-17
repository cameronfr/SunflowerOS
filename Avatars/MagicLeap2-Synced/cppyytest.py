#%%#

# printf stops working after a bit for some reason.
# had problem where compilation would fail without saying why.
# if #include fails the first time, it'll (usually?) fail the same way when try it every subsequent time (have to restart kernel)

# if launch in termux, dlsym after load libpytorch_jni.so works fine. but if launch in dalvik process, dlsym after load libpytorch_jni fails.

import sys
import cppyy
import os
print("Cppyy version:", cppyy.__version__)
cppyy.cppexec("#include <stdio.h>")
cppyy.cppexec("""printf("C++ version %ld", __cplusplus);""")
print("Python version:", sys.version)

# _ZN6caffe28TypeMeta13typeMetaDatasEv is a dynamic symbol in libpytorch_jni.so
# _ZNK8facebook3jni7JBuffer16getDirectAddressEv is a dynamic symbol in libfbjni.so

cppyy.cppdef("""
#include <stdio.h>
#include <stdlib.h>
#include <android/log.h>
// fake printf using android log
#define printf(...) __android_log_print(ANDROID_LOG_INFO, "Sunflower cppyy", __VA_ARGS__)
""")

# cppyy.cppdef("""
# void *java_vm;
# void get_java_vm_from_env() {
#   sscanf(getenv("JAVA_VM_PTR"), "%p", &java_vm);
#   printf("java_vm: %p", java_vm);
# }
# """)
# cppyy.gbl.get_java_vm_from_env()

# cppyy.gbl.gInterpreter.GetIncludePath()

# There is a macro named ClassDef in ROOT that is included in cling and conflicts with ClassDef struct in torch. Have to go to torch header torch/csrc/jit/frontend/tree_views.h, rename ClassDef. (which prob breaks something in libtorch). Or undef ClassDef (which mb breaks something in ROOT).
cppyy.add_include_path("/data/data/com.termux/files/home/MagicLeap2-Synced/libraries/pytorch_headers")
cppyy.add_library_path("/data/data/com.termux/files/home/MagicLeap2-Synced/libraries")
cppyy.cppdef("""
#undef ClassDef
#include <torch/script.h>
""")
cppyy.load_library("libfbjni.so") # either this, or setting rpath on libpytorch_jni.so (when compile to libSunflowerOS.so, rpath is set to ~/MagicLeap2-Synced/libraries during build process. But here, adding library path doesn't affect android linker / dlopen)
cppyy.load_library("libpytorch_jni.so")
# import ctypes
# ctypes.CDLL("/data/data/com.termux/files/home/MagicLeap2-Synced/libraries/libpytorch_jni.so", ctypes.RTLD_GLOBAL) 

cppyy.cppdef("""
#include <dlfcn.h>
void test_torch_23() {
  // dlerror();
  // void* handle = dlopen("/data/data/com.termux/files/home/MagicLeap2-Synced/libraries/libpytorch_jni.so", RTLD_LAZY | RTLD_GLOBAL);
  // printf("dlerror? %s", dlerror());
  // dlsym(handle, "_ZN3c1019UndefinedTensorImpl10_singletonE");
  torch::Tensor tensor = torch::rand({100, 100});
  printf("tensor: %p", tensor.data_ptr());
  //printf("RTLD_DEFAULT is %d", RTLD_DEFAULT);
  // dlerror();
  // dlsym(RTLD_DEFAULT, "_ZTVN5torch8autograd12AutogradMetaE");
  // printf("dlerror? %s", dlerror());
}
""")
cppyy.gbl.test_torch_23()

#%%#

cppyy.cppexec("""
#include <torch/script.h>
int a = 5;
for (int i = 0; i < 10; i++) {
  torch::Tensor tensor = torch::rand({1000, 1000});
  // print mean
  std::cout << tensor.mean().item<float>() << std::endl;
}
""")


# Test torch tensor
cppyy.cppexec("""
int a = 6;
for (int i = 0; i < 11; i++) {
  torch::Tensor tensor = torch::rand({1000, 1000});
  // print mean
  std::cout << tensor.mean().item<float>() << std::endl;
}
""")

cppyy.cppdef("""
#include <stdio.h>
""")
cppyy.cppexec("""
printf("Hello from C++!");
""")

# Throw std exception
cppyy.cppexec("""
throw std::runtime_error("test");
""")

# 1. Initialize StereoKit, get OpenXR running. Load Stereokit as shared lib, compiled with Cmake. Should also see if can redefine function in SK on the fly, w/ change handler (e.g. if change asset loading code, should reload all assets)? No idea what it's going to look like.
# 2. cpp playground where can initialize and render objects. Not sure how to handle frame loop.

# %%
