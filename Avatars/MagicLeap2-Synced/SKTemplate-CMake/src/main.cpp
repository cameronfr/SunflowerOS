#include "camera.h"
#include "PoseModel.cpp" // Only used for definitions, implementation is dlopened
#include "debug.h"

#include<cmath>

#include <stereokit.h>
#include <stereokit_ui.h>
#include <signal.h>
// Android logging include
#include <android/log.h>
#define LOG_TAG "SunflowerOS"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

#include <torch/script.h>

#include <ml_head_tracking.h>
#include <ml_perception.h>
#include <ml_raycast.h>

#include "OneEuroFilter.cpp"
#include "../../StereoKit/StereoKitC/libraries/sokol_time.h"

#include <unistd.h>
#include <dlfcn.h>
#include <jni.h>

JavaVM *android_vm = NULL;
jobject android_activity = NULL;

extern "C" jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    android_vm = vm;
    // Get activity we're running in
    // JNIEnv *env;
    // vm->GetEnv((void **)&env, JNI_VERSION_1_6);
    // jclass activity_thread = env->FindClass("android/app/ActivityThread");
    // jmethodID current_activity_thread = env->GetStaticMethodID(activity_thread, "currentActivityThread", "()Landroid/app/ActivityThread;");
    // jobject at = env->CallStaticObjectMethod(activity_thread, current_activity_thread);
    // jmethodID get_application = env->GetMethodID(activity_thread, "getApplication", "()Landroid/app/Application;");
    // jobject activity_inst = env->CallObjectMethod(at, get_application);
    // android_activity = env->NewGlobalRef(activity_inst);
    LOGD("In JNI_Onload, Activity: %p VM %p", android_activity, android_vm);
    return JNI_VERSION_1_6;
}

extern "C" void JNI_SetActivity(jobject activity) {
    android_activity = activity;
}


using namespace sk;
using namespace torch::indexing;

CameraForCV camera;

template <typename T> T* dlOpenInstantiateClass(const char *libName) {
  printf("Modules: (Re)loading class %s", libName);
  std::string execPath = "/data/data/com.termux/files/home/MagicLeap2-Synced/SKTemplate-CMake";
  std::string libPath = execPath + std::string("/") + std::string(libName);
  void *handle = dlopen(libPath.c_str(), RTLD_LAZY);
  if (!handle) {
      printf("dlopen failed: %s\n", dlerror());
  }
  T* (*create)();
  create = (T* (*)())dlsym(handle, "create");
  if (!create) {
      printf("dlsym failed: %s\n", dlerror());
  }
  return create();
}
template <typename T> void dlOpenDestroyClass(const char *libName, T* instance) {
  printf("Modules: Destroying class %s\n", libName);
  std::string execPath = "/data/data/com.termux/files/home/MagicLeap2-Synced/SKTemplate-CMake";
  std::string libPath = execPath + std::string("/") + std::string(libName);
  void *handle = dlopen(libPath.c_str(), RTLD_LAZY);
  if (!handle) {
      printf("dlopen failed: %s\n", dlerror());
  }
  void (*destroy)(T*);
  destroy = (void (*)(T*))dlsym(handle, "destroy");
  if (!destroy) {
      printf("dlsym failed: %s\n", dlerror());
  }
  destroy(instance);
}

double targetLengthDistCalc(double z1, double z2, double a1, double a2, double a3, double b1, double b2, double b3, double l) {
  // wolframAlpha solve for n, and then converted with Copilot (fixed by me)
  //Equation: ((z_1+n)*a_1 - (z_2+n)*b_1)**2 + ((z_1+n)*a_2 - (z_2+n)*b_2)**2 + ((z_1+n)*a_3 - (z_2+n)*b_3)**2= l**2, solve for n
  // Basically, we have two rays from the camera a_vec and b_vec. We want dist((z_1+n)*a_vec, (z_2+n)*b_vec) = l

  //n = (1/2 sqrt((2 a_1 b_1 z_1 + 2 a_1 b_1 z_2 + 2 a_2 b_2 z_1 + 2 a_3 b_3 z_1 + 2 a_2 b_2 z_2 + 2 a_3 b_3 z_2 - 2 a_1^2 z_1 - 2 a_2^2 z_1 - 2 a_3^2 z_1 - 2 b_1^2 z_2 - 2 b_2^2 z_2 - 2 b_3^2 z_2)^2 - 4 (2 a_1 b_1 + 2 a_2 b_2 + 2 a_3 b_3 - a_1^2 - a_2^2 - a_3^2 - b_1^2 - b_2^2 - b_3^2) (2 a_1 b_1 z_1 z_2 + 2 a_2 b_2 z_1 z_2 + 2 a_3 b_3 z_1 z_2 - a_1^2 z_1^2 - a_2^2 z_1^2 - a_3^2 z_1^2 - b_1^2 z_2^2 - b_2^2 z_2^2 - b_3^2 z_2^2 + l^2)) + a_1 b_1 z_1 + a_1 b_1 z_2 + a_2 b_2 z_1 + a_3 b_3 z_1 + a_2 b_2 z_2 + a_3 b_3 z_2 + a_1^2 (-z_1) - a_2^2 z_1 - a_3^2 z_1 - b_1^2 z_2 - b_2^2 z_2 - b_3^2 z_2)/(-2 a_1 b_1 - 2 a_2 b_2 - 2 a_3 b_3 + a_1^2 + a_2^2 + a_3^2 + b_1^2 + b_2^2 + b_3^2) 
  double n = (0.5*sqrt(
      pow(2*a1*b1*z1 + 2*a1*b1*z2 + 2*a2*b2*z1 + 2*a3*b3*z1 + 2*a2*b2*z2 + 2*a3*b3*z2 - 2*a1*a1*z1 - 2*a2*a2*z1 - 2*a3*a3*z1 - 2*b1*b1*z2 - 2*b2*b2*z2 - 2*b3*b3*z2, 2) 
      - 4*(2*a1*b1 + 2*a2*b2 + 2*a3*b3 - a1*a1 - a2*a2 - a3*a3 - b1*b1 - b2*b2 - b3*b3)*(2*a1*b1*z1*z2 + 2*a2*b2*z1*z2 + 2*a3*b3*z1*z2 - a1*a1*z1*z1 - a2*a2*z1*z1 - a3*a3*z1*z1 - b1*b1*z2*z2 - b2*b2*z2*z2 - b3*b3*z2*z2 + l*l)
      ) + a1*b1*z1 + a1*b1*z2 + a2*b2*z1 + a3*b3*z1 + a2*b2*z2 + a3*b3*z2 + a1*a1*(-z1) - a2*a2*z1 - a3*a3*z1 - b1*b1*z2 - b2*b2*z2 - b3*b3*z2
    )/ (-2*a1*b1 - 2*a2*b2 - 2*a3*b3 + a1*a1 + a2*a2 + a3*a3 + b1*b1 + b2*b2 + b3*b3);

  return n;
}


template <typename T> void print(const char *m, T t) {
  std::ostringstream os;
  os << t << std::endl;
  LOGD("%s %s", m, os.str().c_str());
}

extern int main(int argc, char *argv[]);

void signal_callback_handler(int signum) {
  ALOGD("Caught signal %d", signum);
  sk_quit();
  camera.Destroy();
}

// Not working in forked process, seems to use binder?
void doRaycastTest() {
  // Inits for raycast api
  MLHandle raycast_handle_;
  MLResult mlResult;
  mlResult = MLRaycastCreate(&raycast_handle_);
  if (mlResult != MLResult_Ok) {
    LOGD("MLRaycastCreate failed with error %s", MLGetResultString(mlResult));
  } else if (mlResult == MLResult_Ok) {
    LOGD("MLRaycastCreate success");
  }
  // auto skVecToMlVec = [](vec3 v) {MLVec3f mlV; mlV.x = v.x; mlV.y = v.y; mlV.z = v.z; return mlV;};
  MLRaycastQuery raycast_query = {};
  // raycast_query.position = MLVecml_head_transform.position;
  raycast_query.position = MLVec3f({0,0,0});
  raycast_query.direction = MLVec3f{0,0,-1};
  raycast_query.up_vector = MLVec3f{0,1,0};
  // raycast_query.direction = skVecToMlVec(head_quat * vec3{0,0,-1});
  // raycast_query.up_vector = skVecToMlVec(head_quat * vec3{0,1,0});
  raycast_query.width = 1;
  raycast_query.height = 1;
  raycast_query.horizontal_fov_degrees = 40.0f;
  raycast_query.collide_with_unobserved = true;
  MLHandle raycast;
  mlResult = MLRaycastRequest(raycast_handle_, &raycast_query, &raycast);
  if (mlResult != MLResult_Ok) {
    LOGD("MLRaycastRequest failed with error %d", mlResult);
  } else {
    LOGD("MLRaycastRequest success");
  }
  MLRaycastResult raycast_result;
  mlResult = MLRaycastGetResult(raycast_handle_, raycast, &raycast_result);
  if (mlResult == MLResult_Ok) {
    if (raycast_result.state == MLRaycastResultState_HitObserved) {
      LOGD("Hit at %f %f %f", raycast_result.hitpoint.x, raycast_result.hitpoint.y, raycast_result.hitpoint.z);
    }
  } else if (mlResult == MLResult_Pending) {
    LOGD("MLResult_Pending");
  } else {
    LOGD("MLRaycastGetResult failed with error %d", mlResult);
  }
}

void drawModelWithPose(model_t avatar, torch::Tensor poseWorld, torch::Tensor poseVisibilities, std::unordered_map<model_node_id, matrix> avatarInitialLocalTransforms,  std::unordered_map<model_node_id, matrix> avatarInitialModelTransforms, mesh_t desktop_mesh, material_t desktop_material, mesh_t cube_mesh, material_t cube_mat) {
  int SHOW_POSE_DEBUG = false;
  vec3 head_pos = input_head()->position;
  quat head_quat = input_head()->orientation;

  vec3 hudItemPos = (head_quat * vec3({-0.2, 0.2, -1})) + head_pos;
  if (SHOW_POSE_DEBUG) {
    render_add_mesh(desktop_mesh, desktop_material, pose_matrix({hudItemPos, head_quat}, vec3_one * 1));
  }

  auto poseOnWorldData = poseWorld.accessor<float, 2>();
  if (SHOW_POSE_DEBUG) {
    for (int i = 0; i < poseWorld.size(0); i++) {
      vec3 pos = {poseOnWorldData[i][0], poseOnWorldData[i][1], poseOnWorldData[i][2]};
      render_add_mesh(cube_mesh, cube_mat, pose_matrix({pos, quat_identity}, vec3_one*0.4));
    }
  }

  // GHUM hierarchy
  std::vector <std::array<int, 2>> connLines = {
    {12, 24}, {24, 26}, {26, 28}, {28, 30}, {30, 32}, {11, 23}, {23, 25}, {25, 27}, {27, 29}, {29, 31}, {11, 13}, {13, 15}, {15, 17}, {17, 19}, {15, 21}, {12, 14}, {14, 16}, {16, 18}, {18, 20}, {16, 22}, {23, 24}, {11, 12}
  };
  if (SHOW_POSE_DEBUG) {
    for (int i = 0; i < connLines.size(); i++) {
      vec3 p1 = *reinterpret_cast<vec3*>(poseWorld.index({connLines[i][0], Slice()}).data_ptr());
      vec3 p2 = *reinterpret_cast<vec3*>(poseWorld.index({connLines[i][1], Slice()}).data_ptr());
      float visProbP1 = poseVisibilities[connLines[i][0]].item<float>();
      float visProbP2 = poseVisibilities[connLines[i][1]].item<float>();
      if (std::min(visProbP1, visProbP2) > 0.6) {
        line_add(p1, p2, {200,100,0,100}, {255,120,0,100}, 0.01f);
      }
    }
  }

  auto tensorToVec3 = [](torch::Tensor t) {
    return *reinterpret_cast<vec3*>(t.data_ptr());
  };
  // Note: REQUIRES orthorgonal vectors, or rets NaN in the matrix
  auto quatFromNewBasis = [](vec3 x, vec3 y, vec3 z) {
    x = vec3_normalize(x);
    y = vec3_normalize(y);
    z = vec3_normalize(z);
    matrix m = matrix{ 
      x.x, x.y, x.z, 0, 
      y.x, y.y, y.z, 0, 
      z.x, z.y, z.z, 0, 
      0,   0,   0,   1
    };
    quat q = matrix_extract_rotation(m);
    return q;
  };
  // lambda that returns reference to vec3. Don't use reference, because it gets invalidated at some point(?) not sure
  auto poseWorldVecs = [&](int i) -> vec3 {
    return tensorToVec3(poseWorld.index({i, Slice()}));
  };
  auto quatBetweenVecs = [&](vec3 a, vec3 b) {
    // Shortest rotation from a to b. doesn't do spinning either.
    vec3 axis = vec3_cross(a, b);
    float w = (vec3_magnitude(a) * vec3_magnitude(b)) + vec3_dot(a, b);
    quat q = {axis.x, axis.y, axis.z, w};
    quat qOut = quat_normalize(q);
    return qOut;
  };

  // _t means target, and we'll try to apply these targets to the pose of the avatar
  // Aligning by imagining case where both person and avatar are facing towards camera (but it's actually away from camera?)
  vec3 hipsCenter_t = (poseWorldVecs(23) + poseWorldVecs(24)) / 2;  
  vec3 shouldersCenter_t = (poseWorldVecs(11) + poseWorldVecs(12)) / 2;
  vec3 acrossHips_t = poseWorldVecs(23) - poseWorldVecs(24); // towards +x
  vec3 upHips_t = shouldersCenter_t - hipsCenter_t; // towards +y
  vec3 outHips_t = vec3_cross(acrossHips_t, upHips_t); // towards +z
  vec3 orthogonalHipPlane_t = vec3_cross(outHips_t, acrossHips_t); // towards +y
  // Assumes that w/ hips with localTransform=identity, the model is facing towards +z (i.e. towards camera in SK case)
  quat hipsRot_initial = quat_from_angles(0, 0, 180);
  // std::cout << "acroships_t: " << acrossHips_t.x << " " << acrossHips_t.y << " " << acrossHips_t.z << std::endl;
  // std::cout << "uphips_t: " << upHips_t.x << " " << upHips_t.y << " " << upHips_t.z << std::endl;
  // std::cout << "crosships_t: " << crossHips_t.x << " " << crossHips_t.y << " " << crossHips_t.z << std::endl;
  quat hipsRot = quatFromNewBasis(acrossHips_t, orthogonalHipPlane_t, outHips_t);
  torch::Tensor hipPosTensor = (poseWorld.index({23, Slice()}) + poseWorld.index({24, Slice()})) / 2;
  vec3 hipPos = tensorToVec3(hipPosTensor);

  // TODO: all this is stateful, so have to do following hierarchy from parent to child. also does alot of extra matrix mults, might be fine though.

  // Reset the model to t-pose
  model_node_id spine = model_node_find(avatar, "J_Bip_C_Spine");
  model_node_set_transform_local(avatar, spine, avatarInitialLocalTransforms[spine]);
  model_node_id head = model_node_find(avatar, "J_Bip_C_Head");
  model_node_set_transform_local(avatar, head, avatarInitialLocalTransforms[head]);
  model_node_id leftLowerArm = model_node_find(avatar, "J_Bip_L_LowerArm");
  model_node_set_transform_local(avatar, leftLowerArm, avatarInitialLocalTransforms[leftLowerArm]);
  model_node_id rightLowerArm = model_node_find(avatar, "J_Bip_R_LowerArm");
  model_node_set_transform_local(avatar, rightLowerArm, avatarInitialLocalTransforms[rightLowerArm]);
  model_node_id leftLowerLeg = model_node_find(avatar, "J_Bip_L_LowerLeg");
  model_node_set_transform_local(avatar, leftLowerLeg, avatarInitialLocalTransforms[leftLowerLeg]);
  model_node_id rightLowerLeg = model_node_find(avatar, "J_Bip_R_LowerLeg");
  model_node_set_transform_local(avatar, rightLowerLeg, avatarInitialLocalTransforms[rightLowerLeg]);

  model_node_id hipsNode = model_node_find(avatar, "J_Bip_C_Hips");
  model_node_set_transform_model(avatar, hipsNode, pose_matrix({hipPos + vec3{0, 0.12, 0}, hipsRot_initial * hipsRot}, vec3_one * 1.15));

  auto setNodeRelativeRotationFromTPose = [&](char *nodeName, quat rotation) {
    model_node_id modelNode = model_node_find(avatar, nodeName);
    // Reset to t-pose rotation 
    model_node_set_transform_local(avatar, modelNode, avatarInitialLocalTransforms[modelNode]);
    matrix startTrans = model_node_get_transform_model(avatar, modelNode);
    // Apply our global rotation to joint
    matrix newTrans = matrix_trs(matrix_extract_translation(startTrans), matrix_extract_rotation(startTrans) * rotation, matrix_extract_scale(startTrans));
    model_node_set_transform_model(avatar, modelNode, newTrans);
  };

  vec3 p1 = matrix_extract_translation(model_node_get_transform_model(avatar, model_node_find(avatar, "J_Bip_C_Spine")));
  vec3 p2 = matrix_extract_translation(model_node_get_transform_model(avatar, model_node_find(avatar, "J_Bip_C_Hips")));
  // line_add(p1, p2, {0,0,255,255}, {0,0,255,255}, 0.01f);
  // line_add(p1, p1+upHips_t, {0,255,0,255}, {0,255,0,255}, 0.01f);
  setNodeRelativeRotationFromTPose("J_Bip_C_Spine", quatBetweenVecs(
    p1 - p2,
    upHips_t
  )); // Jank IK

  vec3 noseTop_t = (poseWorldVecs(4) + poseWorldVecs(1)) / 2;
  vec3 mouthCenter_t = (poseWorldVecs(10) + poseWorldVecs(9)) / 2;
  vec3 upFace_t = noseTop_t - mouthCenter_t;
  vec3 acrossFace_t = poseWorldVecs(7) - poseWorldVecs(8);
  vec3 outFace_t = vec3_cross(acrossFace_t, upFace_t);
  vec3 orthog_upFace = vec3_cross(outFace_t, acrossFace_t);
  quat headQuatGlobal = quatFromNewBasis(acrossFace_t, orthog_upFace, outFace_t);

  auto drawBasis = [&](vec3 origin, vec3 x, vec3 y, vec3 z) {
    line_add(origin, origin + 0.1*vec3_normalize(x), {255,0,0,255}, {255,0,0,255}, 0.01f);
    line_add(origin, origin + 0.1*vec3_normalize(y), {0,255,0,255}, {0,255,0,255}, 0.01f);
    line_add(origin, origin + 0.1*vec3_normalize(z), {0,0,255,255}, {0,0,255,255}, 0.01f);
  };
  drawBasis(vec3{0,0,0}, vec3{1,0,0}, vec3{0,1,0}, vec3{0,0,1});

  model_node_set_transform_model(avatar, head, matrix_trs(
    matrix_extract_translation(model_node_get_transform_model(avatar, head)),
    matrix_extract_rotation(avatarInitialModelTransforms[head]) * quat_from_angles(-35, 0, 0) * headQuatGlobal,
    matrix_extract_scale(model_node_get_transform_model(avatar, head))
  )); // more jank ik (put into t-pose rotation facing z-forward in world space, tilt up the head, then apply target rot in world space)

  setNodeRelativeRotationFromTPose("J_Bip_R_UpperArm", quatBetweenVecs(
    poseWorldVecs(12) - poseWorldVecs(11),
    poseWorldVecs(14) - poseWorldVecs(12)
  ));
  // IK rest of the way, trying to match hand position
  vec3 rightLowerArmPos = matrix_extract_translation(model_node_get_transform_model(avatar,rightLowerArm));
  model_node_set_transform_model(avatar, rightLowerArm, matrix_trs(
    rightLowerArmPos,
    matrix_extract_rotation(avatarInitialModelTransforms[rightLowerArm]) * quatBetweenVecs(
      vec3{-1,0,0},
      poseWorldVecs(16) - rightLowerArmPos
    ),
    matrix_extract_scale(model_node_get_transform_model(avatar, rightLowerArm))
  ));
  // line_add(rightLowerArmPos, poseWorldVecs(16), {255,0,0,255}, {255,0,0,255}, 0.01f);

  setNodeRelativeRotationFromTPose("J_Bip_L_UpperArm", quatBetweenVecs(
    poseWorldVecs(11) - poseWorldVecs(12),
    poseWorldVecs(13) - poseWorldVecs(11)
  ));
  // IK rest of the way, trying to match hand position
  vec3 leftLowerArmPos = matrix_extract_translation(model_node_get_transform_model(avatar, leftLowerArm));
  model_node_set_transform_model(avatar, leftLowerArm, matrix_trs(
    leftLowerArmPos,
    matrix_extract_rotation(avatarInitialModelTransforms[leftLowerArm]) * quatBetweenVecs(
      vec3{1,0,0},
      poseWorldVecs(15) - leftLowerArmPos
    ),
    matrix_extract_scale(model_node_get_transform_model(avatar, leftLowerArm))
  ));
  // line_add(leftLowerArmPos, poseWorldVecs(15), {255,0,0,255}, {255,0,0,255}, 0.01f);

  setNodeRelativeRotationFromTPose("J_Bip_R_UpperLeg", quatBetweenVecs(
    -upHips_t,
    poseWorldVecs(26) - poseWorldVecs(24)
  ));
  vec3 rightLowerLegPos = matrix_extract_translation(model_node_get_transform_model(avatar, rightLowerLeg));
  model_node_set_transform_model(avatar, rightLowerLeg, matrix_trs(
    rightLowerLegPos,
    matrix_extract_rotation(avatarInitialModelTransforms[rightLowerLeg]) * quatBetweenVecs(
      vec3{0,-1,0},
      poseWorldVecs(28) - rightLowerLegPos
    ),
    matrix_extract_scale(model_node_get_transform_model(avatar, rightLowerLeg))
  ));
  setNodeRelativeRotationFromTPose("J_Bip_R_LowerLeg", quatBetweenVecs(
    poseWorldVecs(26) - poseWorldVecs(24),
    poseWorldVecs(28) - poseWorldVecs(26)
  ));
  setNodeRelativeRotationFromTPose("J_Bip_L_UpperLeg", quatBetweenVecs(
    -upHips_t,
    poseWorldVecs(25) - poseWorldVecs(23)
  ));
  vec3 leftLowerLegPos = matrix_extract_translation(model_node_get_transform_model(avatar, leftLowerLeg));
  model_node_set_transform_model(avatar, leftLowerLeg, matrix_trs(
    leftLowerLegPos,
    matrix_extract_rotation(avatarInitialModelTransforms[leftLowerLeg]) * quatBetweenVecs(
      vec3{0,-1,0},
      poseWorldVecs(27) - leftLowerLegPos
    ),
    matrix_extract_scale(model_node_get_transform_model(avatar, leftLowerLeg))
  ));
  setNodeRelativeRotationFromTPose("J_Bip_L_LowerLeg", quatBetweenVecs(
    poseWorldVecs(25) - poseWorldVecs(23),
    poseWorldVecs(27) - poseWorldVecs(25)
  ));

  model_draw(avatar,  pose_matrix({{0, 0, 0}, quat_identity}, vec3_one * 1.0));
  // render_add_mesh(cube_mesh, cube_mat, rShoulderGlobalTransform);
}

int main(int argc, char *argv[]) {

  signal(SIGINT, signal_callback_handler);
  sk_settings_t settings = {};
  LOGD("In main, Activity: %p VM %p", android_activity, android_vm);
  settings.android_java_vm = (void *)android_vm;
  settings.android_activity = (void *)android_activity;

	settings.app_name           = "SunflowerOS v0.1";
	settings.assets_folder      = "/data/data/com.termux/files/home/MagicLeap2-Synced/StereoKit/Examples/Assets/";
	settings.display_preference = display_mode_mixedreality;
	// settings.display_preference = display_mode_flatscreen;

  backend_openxr_ext_request("XR_ML_ml2_controller_interaction");
  LOGD("Initializing SK");
	if (!sk_init(settings)) {
    LOGD("SK Init failed");
    return 1;
  }

  // Initialize perception. Required before using things that depend on perception (e.g. camera pose)
  MLPerceptionSettings perception_settings;
  int result;
	result = MLPerceptionInitSettings(&perception_settings);
	if (MLResult_Ok != result) {
		LOGD("MLPerceptionInitSettings failed with %d, exiting app", result);
		return 1;
	}
	result = MLPerceptionStartup(&perception_settings);
	if (MLResult_Ok != result) {
		LOGD("MLPerceptionStartup failed with %d, exiting app", result);
		return 1;
	}

  // Initialize our CV Camera
  camera.Initialize();
  camera.Start();

  // -Z forwards, +Y up, +X right
  // camInfo.transform has same coord system, but Magic leap proj must be changing it
  mesh_t     cube_mesh;
  model_t avatar;
  material_t cube_mat;
  pose_t cube_pose;

  // avatar = model_create_file("../../../vroiddemo.glb");
  // avatar = model_create_file("../../../vroiddemo_manualoptim_1.glb");
  avatar = model_create_file("../../../vroid_manualoptim_1_throughblender.glb");
  // avatar = model_create_file("DamagedHelmet.gltf");
  // avatar = model_create_file("Cosmonaut.glb");
  std::unordered_map<model_node_id, matrix> avatarInitialLocalTransforms;
  std::unordered_map<model_node_id, matrix> avatarInitialModelTransforms;
  for (int i = 0; i < model_node_count(avatar); i++) {
    model_node_id nodeId = model_node_id(i);
    avatarInitialLocalTransforms[nodeId] = model_node_get_transform_local(avatar, nodeId);
    avatarInitialModelTransforms[nodeId] = model_node_get_transform_model(avatar, nodeId);
  }

  OneEuroFilter poseWorldFilter;
  poseWorldFilter.Initialize(0, torch::zeros({35, 3}), torch::zeros({35, 3}), 0.005, 0.03875, 1.0);

  cube_pose = {{0,0,-0.5f}, quat_identity};
  // cube_pose = {{0,-2.5, 0}, quat_identity};
  // cube_pose = {{-2.5,0,0}, quat_identity};

  cube_mesh = mesh_gen_rounded_cube(vec3_one * 0.03f * 1, 0.02f, 4);
  cube_mat  = material_find        (default_id_material);
	mesh_t desktop_mesh = mesh_gen_plane({0.2*1.0, 0.2}, { 0,0,1 }, {0,1,0}); 
  material_t desktop_material = material_copy_id (default_id_material_unlit);
  tex_t cameraTex = tex_create(tex_type_image, tex_format_rgba32);
  material_set_texture(desktop_material, "diffuse", cameraTex);


  int frameNum = 0;
  PoseModel *poseModel = nullptr;
  torch::Tensor poseWorld = torch::zeros({33, 3});
  torch::Tensor poseVisibilities = torch::zeros({33});

  static auto update = [&]() {
    
    if (frameNum % (60*240) == 0) {
      if (poseModel != nullptr) {
        dlOpenDestroyClass("libPoseModel.so", poseModel);
      }
      poseModel = dlOpenInstantiateClass<PoseModel>("libPoseModel.so");
    }
    frameNum++;

    // Update poseWorld
    if (frameNum % 1 == 0) {
      auto cameraOut = camera.GetOutput();
      torch::Tensor cameraImg = std::get<0>(cameraOut);
      MLTransform imgCamTrans = std::get<1>(cameraOut); //must use pose at time img was taken, otherwise output will follow head
      vec3 imgCamTrans_pos = (vec3 &)imgCamTrans.position;
      quat imgCamTrans_quat = (quat &)imgCamTrans.rotation;
      bool foundPerson;
      try {
        // tex_set_colors(cameraTex, 1280, 960, cameraImg.data_ptr());
        // This method must be virtual if want it to be updated on dlopen
        foundPerson = poseModel->ProcessImage(cameraImg, cameraTex);
      } catch (const std::exception& e) {
        LOGD("Exception: %s", e.what());
      }
      if (foundPerson) {
        // Only update poseWorld if found person (otherwise will be moving old pose w/ current head pos)

        // Stereokit uses row-major, local-on-left. GL uses row-major, local-on-right. Torch uses row-major.
        // Local-on-left because DirectX uses version of transform-matrix that looks transpose from standard, and it's like [1x4] * [matrix^T] instead of [4x1] * [matrix]. Stereokit uses DirectX transform-matrix construction functions, so it's local-on-left.
        // See http://davidlively.com/programming/graphics/opengl-matrices/row-major-vs-column-major/
        torch::Tensor poseImageCoords = poseModel->GetLatestPose(); //33 x [x,y,z]
        poseVisibilities = poseModel->GetLatestPoseVisibilities(); //33 x 1]

        // TODO: use the per-frame camera intrinsics in MapImageCoordsToCameraCoords (not sure how much they're changing, mb when autofocus and stuff?)
        torch::Tensor poseImageZ = poseImageCoords.index({Slice(), 2});// * -1; // map coord to -Z forwards
        torch::Tensor poseXYOnCameraPlane = camera.MapImageCoordsTo3DCoords(poseImageCoords.index({Slice(), Slice(0,2)}), 1.0f);
        poseXYOnCameraPlane *= torch::from_blob((float[]){1, -1, -1}, {3}, torch::kFloat); // map coord system　to sk

        // Now, can change z-distance of each of these point in poseXYOnCameraPlane by multiplying each one, which will move it along the ray from the camera to the point
        // each of these rays has z=1, so resulting z will be z_target given by poseImageZScaled
        torch::Tensor poseImageZScaled = poseImageZ.unsqueeze(1) * camera.GetIntrinsicsInverseMatrixXScale();
        torch::Tensor poseRelativeToCamera = poseImageZScaled  * poseXYOnCameraPlane;

        // Choose distance of hip from camera by making the shoulder-width a fixed value. This means that distance will only be accurate if the human in question has that shoulder-width.
        double z1, z2, a1, a2, a3, b1, b2, b3, l;
        // Shoulder dist
        // z1 = poseImageZScaled[11][0].item<double>();
        // z2 = poseImageZScaled[12][0].item<double>();
        // a1 = poseXYOnCameraPlane[11][0].item<double>();
        // a2 = poseXYOnCameraPlane[11][1].item<double>();
        // a3 = poseXYOnCameraPlane[11][2].item<double>();
        // b1 = poseXYOnCameraPlane[12][0].item<double>();
        // b2 = poseXYOnCameraPlane[12][1].item<double>();
        // b3 = poseXYOnCameraPlane[12][2].item<double>();
        // l = 0.31; // 0.31 meter shoulder-distance target. TODO: use avg of bone sizes somehow.
        // Spine length (sorta, just left kpts. This better because if person is standing and turns to side, z-output of model isn't super accurate and will mess up shoulder length, but xy is still accurate and will give good spine length)
        torch::Tensor shoulderCenterZ = (poseImageZScaled[11] + poseImageZScaled[12]) / 2;
        torch::Tensor shoulderCenterXY = (poseXYOnCameraPlane[11] + poseXYOnCameraPlane[12]) / 2;
        torch::Tensor waistCenterZ = (poseImageZScaled[23] + poseImageZScaled[24]) / 2;
        torch::Tensor waistCenterXY = (poseXYOnCameraPlane[23] + poseXYOnCameraPlane[24]) / 2;
        z1 = shoulderCenterZ[0].item<double>();
        z2 = waistCenterZ[0].item<double>();
        a1 = shoulderCenterXY[0].item<double>();
        a2 = shoulderCenterXY[1].item<double>();
        a3 = shoulderCenterXY[2].item<double>();
        b1 = waistCenterXY[0].item<double>();
        b2 = waistCenterXY[1].item<double>();
        b3 = waistCenterXY[2].item<double>();
        l = 0.51; 
        double estDistFromCamera = targetLengthDistCalc(z1, z2, a1, a2, a3, b1, b2, b3, l);
        // float estDistFromCamera = 2.0;
        poseRelativeToCamera += estDistFromCamera *  poseXYOnCameraPlane;
        // log_diagf("estDistFromCamera: %f", estDistFromCamera);
        // float newShoulderDist = torch::norm(poseRelativeToCamera[11] - poseRelativeToCamera[12]).item<float>();
        // log_diagf("Shoulder dist of pose, this should be constant: %f", newShoulderDist);



        matrix physicalCameraToWorldSpaceMat = pose_matrix({imgCamTrans_pos, imgCamTrans_quat}, vec3_one);
        torch::Tensor physicalCameraToWorldSpace = torch::from_blob(physicalCameraToWorldSpaceMat.m, {4,4}, torch::kFloat); // this is the same as the sk matrix and we can use it as [nx4] * [matrix]
        poseWorld = torch::mm(poseRelativeToCamera, physicalCameraToWorldSpace.index({Slice(0,3), Slice(0,3)})) + physicalCameraToWorldSpace.index({3, Slice(0, 3)}); // does same thing as if we made the poseOnCameraPlane points nx4 (by appending one) and multiplied by the 4x4 matrix

        poseWorld = poseWorldFilter.filter(stm_us(stm_now()), poseWorld);
      } else {
        poseWorldFilter.ResetHistory();
      }
    }

    drawModelWithPose(avatar, poseWorld, poseVisibilities, avatarInitialLocalTransforms, avatarInitialModelTransforms, desktop_mesh, desktop_material, cube_mesh, cube_mat);

    {
      static pose_t window_pose = //pose_t{ vec3{1,1,1} * 0.9f, quat_lookat({1,1,1}, {0,0,0}) };
      pose_t{ {0,0,-0.25f}, quat_lookat({0,0,-0.25f}, {0,0,0}) };
      ui_window_begin("Aux Kpt Filter Smoothing", window_pose, vec2{ 24 }*cm2m);
      
      static float min_cutoff = 0.6;
      static float beta = 0.8;
      static float d_cutoff = 1.0;
      ui_hslider("min_cutoff", min_cutoff, 0.00f, 0.8f, 0, 140 * mm2m); 
      ui_sameline();
      char *min_cutoff_str = (char *)alloca(32);
      sprintf(min_cutoff_str, "min_cutoff: %.4f", min_cutoff);
      ui_text(min_cutoff_str);

      ui_nextline();
      ui_hslider("beta", beta, 0.00f, 0.8f, 0, 140 * mm2m); 
      ui_sameline();
      char *beta_str = (char *)alloca(32);
      sprintf(beta_str, "beta: %.4f", beta);
      ui_text(beta_str);

      ui_nextline();
      ui_hslider("d_cutoff", d_cutoff, 0.01f, 2.0f, 0, 140 * mm2m); 
      ui_sameline();
      char *d_cutoff_str = (char *)alloca(32);
      sprintf(d_cutoff_str, "d_cutoff: %.4f", d_cutoff);
      ui_text(d_cutoff_str);
      // poseModel->aux_keypoints_filter.ChangeParams(min_cutoff, beta, d_cutoff);
      poseWorldFilter.ChangeParams(min_cutoff, beta, d_cutoff);
      ui_window_end();
    }
  };
  auto update_ptr = []() { update(); };

  sk_run(update_ptr);

  return 0;
}
