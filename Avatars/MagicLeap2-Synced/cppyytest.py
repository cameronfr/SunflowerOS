#%%#

# Notes

# printf stops working after a bit for some reason.
# had problem where compilation would fail without saying why.
# if #include fails the first time, it'll (usually?) fail the same way when try it every subsequent time (have restart kernel and add the include path)
# if do "using namespace sk" once in cppdef, can't undo
# Kernel interrupt doesn't work because interrupt is sent to "fake" kernel process

# 1. Initialize StereoKit, get OpenXR running. Load Stereokit as shared lib, compiled with Cmake. Should also see if can redefine function in SK on the fly, w/ change handler (e.g. if change asset loading code, should reload all assets)? No idea what it's going to look like.
# 2. cpp playground where can initialize and render objects. Not sure how to handle frame loop.

#%%# IMPORTS

import sys
import cppyy
import os
print("Cppyy version:", cppyy.__version__)
cppyy.cppexec("#include <stdio.h>")
cppyy.cppexec("""printf("C++ version %ld", __cplusplus);""")
print("Python version:", sys.version)
print("OS Name: ", os.uname().release)

cppyy.cppdef("""
#include <stdio.h>
#include <stdlib.h>
#include <android/log.h>
// fake printf using android log
#define printf(...) __android_log_print(ANDROID_LOG_INFO, "Sunflower cppyy", __VA_ARGS__)
""")


# # Torch
# # There is a macro named ClassDef in ROOT that is included in cling and conflicts with ClassDef struct in torch. undef it. mb this will break something in root.
# cppyy.add_include_path(os.path.expanduser("~/MagicLeap2-Synced/libraries/pytorch_headers"))
# cppyy.add_library_path(os.path.expanduser("~/MagicLeap2-Synced/libraries"))
# cppyy.cppdef("""
# #undef ClassDef
# #include <torch/script.h>
# """)
# cppyy.load_library("libfbjni.so") # either this, or setting rpath on libpytorch_jni.so (when compile to libSunflowerOS.so, rpath is set to ~/MagicLeap2-Synced/libraries during build process. But here, adding library path doesn't affect android linker / dlopen)
# cppyy.load_library("libpytorch_jni.so")
# cppyy.cppdef("""
# #include <torch/script.h>
# void test_torch_26() {
#   torch::Tensor tensor = torch::rand({100, 100});
#   printf("tensor: %f", tensor.mean().item<float>());
# }
# """)
# cppyy.gbl.test_torch_26()

# StereoKit
cppyy.add_include_path(os.path.expanduser("~/MagicLeap2-Synced/StereoKit/StereoKitC"))
cppyy.add_library_path(os.path.expanduser("~/MagicLeap2-Synced/StereoKit/"))
cppyy.load_library("libStereoKitC.so")
cppyy.cppdef("""
#include <stereokit.h>
""")


#%%$ Moonlight

# cppyy.add_include_path(os.path.expanduser("~/MagicLeap2-Synced/moonlight-common-c/src/"))
# cppyy.cppdef("""
# #include <Limelight.h>
# """)
# # initialize Moonlight client
# cppyy.cppdef("""
# // Start the Moonlight client
# void start_moonlight() {
  



#%%# StereoKit

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

cppyy.cppdef("""
using namespace sk;

bool initialize_stereokit() {
  sk_settings_t settings = {};
  settings.android_java_vm = java_vm;
  settings.android_activity = (void *)java_android_activity; 

  settings.app_name           = "SunflowerOS cppyy v0.1";
  settings.assets_folder      = "/data/data/com.termux/files/home/MagicLeap2-Synced/StereoKit/Examples/Assets/";
  settings.display_preference = display_mode_mixedreality;

  backend_openxr_ext_request("XR_ML_ml2_controller_interaction");
  printf("Initializing SK");
  if (!sk_init(settings)) {
    printf("SK Init failed");
    return false;
  }
  return true;
}
""")
cppyy.gbl.initialize_stereokit()


cppyy.cppdef("""
namespace tmp_5 {
  pose_t cube_pose;
  mesh_t cube_mesh;
  material_t cube_mat;

  model_t avatar;

  void setup() {
    cube_pose = {{0,0,-0.5f}, quat_identity};
    cube_mesh = mesh_gen_rounded_cube(vec3_one * 1 * 1, 0.02f, 4);
    cube_mat  = material_find(default_id_material);
    avatar = model_create_file("/data/data/com.termux/files/home/MagicLeap2-Synced/vroiddemo_throughblender.glb");
  }

  void update_fn() {
    render_add_mesh(cube_mesh, cube_mat, pose_matrix(cube_pose, vec3_one));
    model_draw(avatar,  pose_matrix({{0, 0, 0}, quat_identity}, vec3_one * 1.0));
  }

  void update() {
    sk_step(update_fn);
    //sk_run(update_fn);
  }
}
""")

cppyy.gbl.tmp_5.setup()
# %%time
for i in range(1000):
  cppyy.gbl.tmp_5.update()




