#include "camera.h"
#include "PoseModel.cpp" // Only used for definitions, implementation is dlopened
#include "debug.h"

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
  avatar = model_create_file("../../../vroiddemo_manualoptim_1.glb");
  // avatar = model_create_file("DamagedHelmet.gltf");
  // avatar = model_create_file("Cosmonaut.glb");

  // LOGD("\nAnimation count: %d", model_anim_count(avatar));
  // for(int i = 0; i < model_anim_count(avatar); i++) {
  //     LOGD("\nAnimation %d: %s", i, model_anim_get_name(avatar, i));
  // }


  // model_play_anim

  cube_pose = {{0,0,-0.5f}, quat_identity};
  // cube_pose = {{0,-2.5, 0}, quat_identity};
  // cube_pose = {{-2.5,0,0}, quat_identity};
  cube_mesh = mesh_gen_rounded_cube(vec3_one * 0.03f, 0.02f, 4);
  cube_mat  = material_find        (default_id_material);

  tex_t cameraTex = tex_create(tex_type_image, tex_format_rgba32);
  material_t desktop_material = material_copy_id (default_id_material_unlit);
  material_set_texture(desktop_material, "diffuse", cameraTex);
	mesh_t desktop_mesh = mesh_gen_plane({0.2, 0.2}, { 0,0,1 }, {0,1,0}); 

  // Inits for magic leap head tracking api
  MLHandle head_tracker_;
  MLHeadTrackingStaticData head_static_data_;
  MLHeadTrackingCreate(&head_tracker_);
  MLHeadTrackingGetStaticData(head_tracker_, &head_static_data_);



  PoseModel *poseModel = nullptr;
  int frameNum = 0;

  torch::Tensor poseWorld = torch::zeros({33, 3});

  static auto update = [&]() {
    if (frameNum % (60*60) == 0) {
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

    model_draw(avatar,  pose_matrix({{0,-0.8,-2.5f}, quat_identity}, vec3_one * 0.99f));
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

        // TODO: probably want camera pos + rotation, not head pos. Also, clean up the camera->world transform math
        // TODO: only update position when have new pose. On off-frames, keep last world-position (don't follow head)
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
        // Put hips at 2.0, then move along ray by Z-distance from model
        poseXYOnCameraPlane = (2.0f + (poseImageZ.unsqueeze(1) * camera.GetIntrinsicsInverseMatrixXScale())) * poseXYOnCameraPlane;

        matrix physicalCameraToWorldSpaceMat = pose_matrix({imgCamTrans_pos, imgCamTrans_quat}, vec3_one);
        torch::Tensor physicalCameraToWorldSpace = torch::from_blob(physicalCameraToWorldSpaceMat.m, {4,4}, torch::kFloat); // this is the same as the sk matrix and we can use it as [nx4] * [matrix]
        poseWorld = torch::mm(poseXYOnCameraPlane, physicalCameraToWorldSpace.index({Slice(0,3), Slice(0,3)})) + physicalCameraToWorldSpace.index({3, Slice(0, 3)}); // does same thing as if we made the poseOnCameraPlane points nx4 (by appending one) and multiplied by the 4x4 matrix
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
