#include "camera.h"
#include "PoseModel.cpp" // Only used for definitions, implementation is dlopened
#include "debug.h"

#include<cmath>

#include <stereokit.h>
#include <stereokit_ui.h>
#include <signal.h>
// Android logging include
#include <android/log.h>
// #include <jni.h>
#define LOG_TAG "SunflowerOS"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

#include <torch/script.h>

#include <ml_head_tracking.h>
#include <ml_perception.h>
#include <ml_raycast.h>

#include <dlfcn.h>


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

float targetLengthDistCalc(float z1, float z2, float a1, float a2, float a3, float b1, float b2, float b3, float l) {
  // wolframAlpha solve for n, and then converted with Copilot (fixed by me)
  //Equation: ((z_1+n)*a_1 - (z_2+n)*b_1)**2 + ((z_1+n)*a_2 - (z_2+n)*b_2)**2 + ((z_1+n)*a_3 - (z_2+n)*b_3)**2= l**2, solve for n
  // Basically, we have two rays from the camera a_vec and b_vec. We want dist((z_1+n)*a_vec, (z_2+n)*b_vec) = l

  //n = (1/2 sqrt((2 a_1 b_1 z_1 + 2 a_1 b_1 z_2 + 2 a_2 b_2 z_1 + 2 a_3 b_3 z_1 + 2 a_2 b_2 z_2 + 2 a_3 b_3 z_2 - 2 a_1^2 z_1 - 2 a_2^2 z_1 - 2 a_3^2 z_1 - 2 b_1^2 z_2 - 2 b_2^2 z_2 - 2 b_3^2 z_2)^2 - 4 (2 a_1 b_1 + 2 a_2 b_2 + 2 a_3 b_3 - a_1^2 - a_2^2 - a_3^2 - b_1^2 - b_2^2 - b_3^2) (2 a_1 b_1 z_1 z_2 + 2 a_2 b_2 z_1 z_2 + 2 a_3 b_3 z_1 z_2 - a_1^2 z_1^2 - a_2^2 z_1^2 - a_3^2 z_1^2 - b_1^2 z_2^2 - b_2^2 z_2^2 - b_3^2 z_2^2 + l^2)) + a_1 b_1 z_1 + a_1 b_1 z_2 + a_2 b_2 z_1 + a_3 b_3 z_1 + a_2 b_2 z_2 + a_3 b_3 z_2 + a_1^2 (-z_1) - a_2^2 z_1 - a_3^2 z_1 - b_1^2 z_2 - b_2^2 z_2 - b_3^2 z_2)/(-2 a_1 b_1 - 2 a_2 b_2 - 2 a_3 b_3 + a_1^2 + a_2^2 + a_3^2 + b_1^2 + b_2^2 + b_3^2) 
  float n = (0.5*sqrt(
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

int main(int argc, char *argv[]) {

  // torch::Tensor a = at::ones({2, 2}, at::kInt);
  // torch::Tensor b = at::randn({2, 2});
  // auto c = a + a + b.to(at::kInt);
  // print("t:", b);

  signal(SIGINT, signal_callback_handler);
  sk_settings_t settings = {};
	settings.app_name           = "SunflowerOS v0.1";
	settings.assets_folder      = "/data/data/com.termux/files/home/MagicLeap2-Synced/StereoKit/Examples/Assets/";
	// settings.display_preference = display_mode_mixedreality;
	settings.display_preference = display_mode_flatscreen;

	if (!sk_init(settings)) return 1;

  // get sk_get_settings fn
  // sk_settings_t (*sk_get_settings)() = (sk_settings_t (*)())dlsym(RTLD_DEFAULT, "sk_get_settings");
  // sk_settings_t test = sk_get_settings();
  // printf("test: %s\n", test.app_name);

  // Initialize camera
  camera.Initialize();
  LOGD("Camera initialized");
  camera.Start();
  LOGD("Camera started");  

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


  // LOGD("\nAnimation count: %d", model_anim_count(avatar));
  // for(int i = 0; i < model_anim_count(avatar); i++) {
  //     LOGD("\nAnimation %d: %s", i, model_anim_get_name(avatar, i));
  // }


  // model_play_anim

  cube_pose = {{0,0,-0.5f}, quat_identity};
  // cube_pose = {{0,-2.5, 0}, quat_identity};
  // cube_pose = {{-2.5,0,0}, quat_identity};
  cube_mesh = mesh_gen_rounded_cube(vec3_one * 0.03f * 1, 0.02f, 4);
  cube_mat  = material_find        (default_id_material);

  tex_t cameraTex = tex_create(tex_type_image, tex_format_rgba32);
  material_t desktop_material = material_copy_id (default_id_material_unlit);
  material_set_texture(desktop_material, "diffuse", cameraTex);
	mesh_t desktop_mesh = mesh_gen_plane({0.2*1.0, 0.2}, { 0,0,1 }, {0,1,0}); 

  // Inits for magic leap head tracking api
  MLHandle head_tracker_;
  MLHeadTrackingStaticData head_static_data_;
  MLHeadTrackingCreate(&head_tracker_);
  MLHeadTrackingGetStaticData(head_tracker_, &head_static_data_);

  PoseModel *poseModel = nullptr;
  int frameNum = 0;

  torch::Tensor poseWorld = torch::zeros({33, 3});

  static auto update = [&]() {
    if (frameNum % (60*240) == 0) {
      if (poseModel != nullptr) {
        dlOpenDestroyClass("libPoseModel.so", poseModel);
      }
      poseModel = dlOpenInstantiateClass<PoseModel>("libPoseModel.so");
    }
    frameNum++;

    // Get head transform from MagicLeap
    MLSnapshot *snapshot = nullptr;
    MLPerceptionGetSnapshot(&snapshot);
    MLTransform ml_head_transform = {};
    MLSnapshotGetTransform(snapshot, &head_static_data_.coord_frame_head, &ml_head_transform);
    MLPerceptionReleaseSnapshot(snapshot);
    vec3 head_pos = (vec3 &)ml_head_transform.position;
    quat head_quat = (quat &)ml_head_transform.rotation;

    // ui_handle_begin("Cube", cube_pose, mesh_get_bounds(cube_mesh), false);
    // ui_handle_end();

    vec3 hudItemPos = (head_quat * vec3({-0.2, 0.2, -1})) + head_pos;
    render_add_mesh(desktop_mesh, desktop_material, pose_matrix({hudItemPos, head_quat}, vec3_one * 1));

    // Update poseWorld
    if (frameNum % 2 == 0) {
      auto cameraOut = camera.GetOutput();
      torch::Tensor cameraImg = std::get<0>(cameraOut);
      MLTransform imgCamTrans = std::get<1>(cameraOut); //must use pose at time img was taken, otherwise output will follow head
      vec3 imgCamTrans_pos = (vec3 &)imgCamTrans.position;
      quat imgCamTrans_quat = (quat &)imgCamTrans.rotation;
      bool foundPerson;
      try {
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

        torch::Tensor poseImageZ = poseImageCoords.index({Slice(), 2});// * -1; // map coord to -Z forwards
        torch::Tensor poseXYOnCameraPlane = camera.MapImageCoordsTo3DCoords(poseImageCoords.index({Slice(), Slice(0,2)}), 1.0f);
        poseXYOnCameraPlane *= torch::from_blob((float[]){1, -1, -1}, {3}, torch::kFloat); // map coord system　to sk

        // Now, can change z-distance of each of these point in poseXYOnCameraPlane by multiplying each one, which will move it along the ray from the camera to the point
        // tensorInfo(poseImageZ, "poseImageZ");
        // tensorInfo(poseXYOnCameraPlane, "poseXYOnCameraPlane");
        // std::cout << "poseXYOnCameraPlane: " << 1.0f + (poseImageZ.unsqueeze(1) * camera.GetIntrinsicsInverseMatrixXScale()) << std::endl;
        // Put hips at 0, and for each point move along ray by Z-distance (which we get from pose model, and is supposed to be approx same scale as x-pixels, and has z=0 at hips)
        torch::Tensor poseImageZScaled = poseImageZ.unsqueeze(1) * camera.GetIntrinsicsInverseMatrixXScale();
        torch::Tensor poseRelativeToCamera = (0.0 + (poseImageZScaled)) * poseXYOnCameraPlane;

        // Choose distance of hip from camera by making the shoulder-width a fixed value. This means that distance will only be accurate if the human in question has that shoulder-width.
        float z1, z2, a1, a2, a3, b1, b2, b3, l;
        z1 = poseImageZScaled[11][0].item<float>();
        z2 = poseImageZScaled[12][0].item<float>();
        a1 = poseXYOnCameraPlane[11][0].item<float>();
        a2 = poseXYOnCameraPlane[11][1].item<float>();
        a3 = poseXYOnCameraPlane[11][2].item<float>();
        b1 = poseXYOnCameraPlane[12][0].item<float>();
        b2 = poseXYOnCameraPlane[12][1].item<float>();
        b3 = poseXYOnCameraPlane[12][2].item<float>();
        l = 0.31; // 0.31 meter shoulder-distance target
        float estDistFromCamera = targetLengthDistCalc(z1, z2, a1, a2, a3, b1, b2, b3, l);
        // float estDistFromCamera = 2.0;

        poseRelativeToCamera += estDistFromCamera * poseXYOnCameraPlane;
        // float newShoulderDist = torch::norm(poseRelativeToCamera[11] - poseRelativeToCamera[12]).item<float>();
        // std::cout << "Shoulder dist of pose, this should be constant" << newShoulderDist << std::endl;

        matrix physicalCameraToWorldSpaceMat = pose_matrix({imgCamTrans_pos, imgCamTrans_quat}, vec3_one);
        torch::Tensor physicalCameraToWorldSpace = torch::from_blob(physicalCameraToWorldSpaceMat.m, {4,4}, torch::kFloat); // this is the same as the sk matrix and we can use it as [nx4] * [matrix]
        poseWorld = torch::mm(poseRelativeToCamera, physicalCameraToWorldSpace.index({Slice(0,3), Slice(0,3)})) + physicalCameraToWorldSpace.index({3, Slice(0, 3)}); // does same thing as if we made the poseOnCameraPlane points nx4 (by appending one) and multiplied by the 4x4 matrix
      }
    }

    auto poseOnWorldData = poseWorld.accessor<float, 2>();
    for (int i = 0; i < poseWorld.size(0); i++) {
      vec3 pos = {poseOnWorldData[i][0], poseOnWorldData[i][1], poseOnWorldData[i][2]};
      render_add_mesh(cube_mesh, cube_mat, pose_matrix({pos, quat_identity}, vec3_one));
    }

    // GHUM hierarchy
    std::vector <std::array<int, 2>> connLines = {
      {12, 24}, {24, 26}, {26, 28}, {28, 30}, {30, 32}, {11, 23}, {23, 25}, {25, 27}, {27, 29}, {29, 31}, {11, 13}, {13, 15}, {15, 17}, {17, 19}, {15, 21}, {12, 14}, {14, 16}, {16, 18}, {18, 20}, {16, 22}, {23, 24}, {11, 12}
    };
    for (int i = 0; i < connLines.size(); i++) {
      vec3 p1 = *reinterpret_cast<vec3*>(poseWorld.index({connLines[i][0], Slice()}).data_ptr());
      vec3 p2 = *reinterpret_cast<vec3*>(poseWorld.index({connLines[i][1], Slice()}).data_ptr());
      line_add(p1, p2, {200,100,0,255}, {200,100,0,255}, 0.01f);
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
    model_node_id hipsNode = model_node_find(avatar, "J_Bip_C_Hips");
    model_node_set_transform_model(avatar, hipsNode, pose_matrix({hipPos + vec3{0, 0.12, 0}, hipsRot_initial * hipsRot}, vec3_one * 1.2));

    auto setNodeRelativeRotationFromTPose = [&](char *nodeName, quat rotation) {
      model_node_id modelNode = model_node_find(avatar, nodeName);
      // Reset to t-pose rotation 
      model_node_set_transform_local(avatar, modelNode, avatarInitialLocalTransforms[modelNode]);
      matrix startTrans = model_node_get_transform_model(avatar, modelNode);
      // Apply our global rotation to joint
      matrix newTrans = matrix_trs(matrix_extract_translation(startTrans), matrix_extract_rotation(startTrans) * rotation, matrix_extract_scale(startTrans));
      model_node_set_transform_model(avatar, modelNode, newTrans);
    };
    setNodeRelativeRotationFromTPose("J_Bip_R_UpperArm", quatBetweenVecs(
      poseWorldVecs(12) - poseWorldVecs(11),
      poseWorldVecs(14) - poseWorldVecs(12)
    ));
    setNodeRelativeRotationFromTPose("J_Bip_R_LowerArm", quatBetweenVecs(
      poseWorldVecs(14) - poseWorldVecs(12),
      poseWorldVecs(16) - poseWorldVecs(14)
    ));
    setNodeRelativeRotationFromTPose("J_Bip_L_UpperArm", quatBetweenVecs(
      poseWorldVecs(11) - poseWorldVecs(12),
      poseWorldVecs(13) - poseWorldVecs(11)
    ));
    setNodeRelativeRotationFromTPose("J_Bip_L_LowerArm", quatBetweenVecs(
      poseWorldVecs(13) - poseWorldVecs(11),
      poseWorldVecs(15) - poseWorldVecs(13)
    ));
    setNodeRelativeRotationFromTPose("J_Bip_R_UpperLeg", quatBetweenVecs(
      -upHips_t,
      poseWorldVecs(26) - poseWorldVecs(24)
    ));
    setNodeRelativeRotationFromTPose("J_Bip_R_LowerLeg", quatBetweenVecs(
      poseWorldVecs(26) - poseWorldVecs(24),
      poseWorldVecs(28) - poseWorldVecs(26)
    ));
    setNodeRelativeRotationFromTPose("J_Bip_L_UpperLeg", quatBetweenVecs(
      -upHips_t,
      poseWorldVecs(25) - poseWorldVecs(23)
    ));
    setNodeRelativeRotationFromTPose("J_Bip_L_LowerLeg", quatBetweenVecs(
      poseWorldVecs(25) - poseWorldVecs(23),
      poseWorldVecs(27) - poseWorldVecs(25)
    ));

    // line_add(poseWorldVecs(12), testVec1+poseWorldVecs(12), {200,0,0,255}, {200,0,0,255}, 0.01f);
    // line_add(poseWorldVecs(12), testVec2+poseWorldVecs(12), {10,255,0,255}, {10,255,0,255}, 0.02f);
    // line_add(poseWorldVecs(12), testVec3+poseWorldVecs(12), {200,255,0,255}, {200,255,0,255}, 0.02f);


    model_draw(avatar,  pose_matrix({{0, 0, 0}, quat_identity}, vec3_one * 1.0));
    // render_add_mesh(cube_mesh, cube_mat, rShoulderGlobalTransform);

    // if (frameNum == 1) {
    //   model_node_id nodeId = 0;
    //   std::cout << "\n\nTraversing";
    //   std::cout << model_node_get_name(avatar, 0) << std::endl;
    //   while (nodeId != -1) {
    //     nodeId = model_node_iterate(avatar, nodeId);
    //     if (nodeId != -1) {
    //       std::cout << model_node_get_name(avatar, nodeId) << std::endl;
    //       matrix localTransform = model_node_get_transform_local(avatar, nodeId);
    //       // torch::Tensor localTransformTorch = torch::from_blob(localTransform.m, {4,4}, torch::kFloat);
    //       // std::cout << localTransformTorch << std::endl;
    //     }
    //   }
    // }



    // tex_set_colors(cameraTex, 1280, 960, camera.GetOutput().data_ptr());
    // render_add_mesh(desktop_mesh, desktop_material, pose_matrix({2,+0.8,-2.5f}, vec3_one * 1));
    // render_add_mesh(desktop_mesh, desktop_material, pose_matrix({4,+0.4,-2.5f}, vec3_one * 1));
    // render_add_mesh(desktop_mesh, desktop_material, pose_matrix({0,+0.8,-2.5f}, vec3_one * 1));
    // render_add_mesh(desktop_mesh, desktop_material, pose_matrix({6,+0.6,-2.5f}, vec3_one * 1));
  };
  auto update_ptr = []() { update(); };

  sk_run(update_ptr);

  return 0;
}
