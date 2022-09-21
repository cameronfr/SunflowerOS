#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <mutex>
#include <vector>



#include <ml_camera_v2.h>
#include <ml_cv_camera.h>
#include <ml_input.h>
#include <ml_head_tracking.h>
#include <ml_media_error.h>
#include <ml_media_format.h>
#include <ml_media_recorder.h>

#include <android/log.h>
#define LOG_TAG "SunflowerOS"
#define ALOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#include <ATen/ATen.h>

using namespace torch::indexing;

namespace EnumHelpers
{
  const char *GetMLCameraErrorString(const MLCameraError &err)
  {
    switch (err)
    {
    case MLCameraError::MLCameraError_None:
      return "";
    case MLCameraError::MLCameraError_Invalid:
      return "Invalid/Unknown error";
    case MLCameraError::MLCameraError_Disabled:
      return "Camera disabled";
    case MLCameraError::MLCameraError_DeviceFailed:
      return "Camera device failed";
    case MLCameraError::MLCameraError_ServiceFailed:
      return "Camera service failed";
    case MLCameraError::MLCameraError_CaptureFailed:
      return "Capture failed";
    default:
      return "Invalid MLCameraError value!";
    }
  }

  const char *GetMLCameraDisconnectReasonString(const MLCameraDisconnectReason &reason)
  {
    switch (reason)
    {
    case MLCameraDisconnectReason::MLCameraDisconnect_DeviceLost:
      return "Device lost";
    case MLCameraDisconnectReason::MLCameraDisconnect_PriorityLost:
      return "Priority lost";
    default:
      return "Invalid MLCameraDisconnectReason value!";
    }
  }
} // namespace EnumHelpers

using namespace std::chrono_literals;

class CameraForCV
{
public:
  CameraForCV()
      : recorder_camera_device_available_(false),
        has_recording_started_(false),
        capture_width_(0),
        capture_height_(0),
        recorder_camera_context_(ML_INVALID_HANDLE),
        capture_fps_(MLCameraCaptureFrameRate_30FPS) {
    
    intrinsic_inverse_matrix = torch::zeros({3, 3});
   }

  void Initialize()
  {
    SetupRestrictedResources();
  }

  void Destroy()
  {
    StopCapture();
    DestroyCamera();
  }

  void Start() 
  {
    MLResult res;
    res = StartCamera();
    if (res != MLResult_Ok)
    {
      ALOGE("Failed to start camera: %s", MLGetResultString(res));
    }
  }

  // Gets image and the pose of the head/camera at the time the img was taken.
  const std::tuple<torch::Tensor, MLTransform> GetOutput() 
  {
    torch::Tensor img = torch::from_blob((void*)framebuffer_.data(), {capture_height_, capture_width_, 4}, torch::kByte); 
    return std::make_tuple(img, current_transform);
  }

  // camera_coords format is N x [x,y] coords on image. maps coords on image plane to coords on plane in camera space, at specified zDistance.
  torch::Tensor MapImageCoordsTo3DCoords(torch::Tensor camera_coords, float zDistance){
    // [x, y, 1]
    torch::Tensor mappedCoords = torch::zeros({3, camera_coords.size(0)}); // 3xN
    mappedCoords.index_put_({Slice(0, 2), Slice()}, camera_coords.permute({1, 0}));  
    mappedCoords.index_put_({2, Slice()}, 1);
    // each column should be a 3x1 vector
    mappedCoords = intrinsic_inverse_matrix.matmul(mappedCoords);
    mappedCoords *= zDistance;

    return mappedCoords.permute({1, 0}); // return Nx [x,y,zDistance] coords
  }

  float GetIntrinsicsInverseMatrixXScale(){
    return intrinsic_inverse_matrix[0][0].item<float>();
  }

private:
  void SetupRestrictedResources()
  {
    MLResult res;
    res = SetupCamera();
    if (res != MLResult_Ok)
    {
      ALOGE("Failed to setup camera! %s", MLGetResultString(res));
    }
    res = SetupCaptureSize();
    if (res != MLResult_Ok)
    {
      ALOGE("Failed to setup camera! %s", MLGetResultString(res));
    }
  }

  void SetIntrinsicMatrixFromMLIntrinsics(const MLCameraIntrinsicCalibrationParameters &intrinsics)
  {
    // TODO: distortion w/ inverse
    torch::Tensor intrinsic_matrix = torch::zeros({3, 3});
    intrinsic_matrix[0][0] = intrinsics.focal_length.x;
    intrinsic_matrix[1][1] = intrinsics.focal_length.y;
    intrinsic_matrix[0][2] = intrinsics.principal_point.x;
    intrinsic_matrix[1][2] = intrinsics.principal_point.y;

    // Scale by the capture size we chose, since ML returned params all seem to be for 2048x1536
    float scaleFactor = (float)capture_width_ / 2048.0f;
    intrinsic_matrix *= scaleFactor;
    intrinsic_matrix[2][2] = 1;

    // Pytorch mobile no .inverse() :/
    intrinsic_inverse_matrix[0][0] = 1.0f / intrinsic_matrix[0][0];
    intrinsic_inverse_matrix[1][1] = 1.0f / intrinsic_matrix[1][1];
    intrinsic_inverse_matrix[0][2] = -intrinsic_matrix[0][2] / intrinsic_matrix[0][0];
    intrinsic_inverse_matrix[1][2] = -intrinsic_matrix[1][2] / intrinsic_matrix[1][1];
    intrinsic_inverse_matrix[2][2] = 1.0f;

    std::cout << "Camera intrinsics inverse matrix: " << intrinsic_inverse_matrix << std::endl;
    // std::cout << "Camera intrinsics: " << intrinsic_matrix << std::endl;
    // float testVecArr[] = {1280, 960};
    // torch::Tensor testVec = torch::from_blob(testVecArr, {1, 2}, torch::kFloat).expand({10, 2});
    // torch::Tensor testPos = MapImageCoordsTo3DCoords(testVec, 1);
    // std::cout << "testPos: \n" << testPos << std::endl;
  }

  static void OnVideoAvailable(const MLCameraOutput *output, const MLHandle metadata_handle,
                               const MLCameraResultExtras *extra, void *data) {
    CameraForCV *this_app = reinterpret_cast<CameraForCV *>(data);
    memcpy(this_app->framebuffer_.data(), output->planes[0].data, output->planes[0].size);
    int res = MLCVCameraGetFramePose(this_app->cv_camera_tracker_, this_app->head_tracker_, MLCVCameraID_ColorCamera, extra->vcam_timestamp, &this_app->current_transform);
    if (res != MLResult_Ok) {
      ALOGE("Failed to get frame pose: %s", MLGetResultString(res));
    }
  }

  MLResult StartCamera() {
    MLResult res;
    MLHandle metadata_handle = ML_INVALID_HANDLE;
    MLCameraCaptureConfig config = {};
    MLCameraCaptureConfigInit(&config);
    framebuffer_ = std::vector<uint8_t>(capture_width_ * capture_height_ * 4);
    config.stream_config[0].capture_type = MLCameraCaptureType_Video;
    config.stream_config[0].width = capture_width_;
    config.stream_config[0].height = capture_height_;
    config.stream_config[0].output_format = MLCameraOutputFormat_RGBA_8888;
    config.stream_config[0].native_surface_handle = ML_INVALID_HANDLE;
    config.capture_frame_rate = capture_fps_;
    config.num_streams = 1;
    res = MLCameraPrepareCapture(recorder_camera_context_, &config, &metadata_handle);
    if (res != MLResult_Ok) {ALOGD("PrepareCapture fail"); return res;}
    res = MLCameraPreCaptureAEAWB(recorder_camera_context_);
    if (res != MLResult_Ok) {ALOGD("AEWB fail"); return res;}
    res = MLCameraCaptureVideoStart(recorder_camera_context_);
    if (res != MLResult_Ok) {ALOGD("CaptureStart fail"); return res;}
    current_capture_len_ms_ = 0;
    has_recording_started_ = true;

    MLCameraIntrinsicCalibrationParameters camera_intrinsics;
    MLCameraIntrinsicCalibrationParametersInit(&camera_intrinsics);
    MLCameraGetIntrinsicCalibrationParameters(recorder_camera_context_, &camera_intrinsics);
    SetIntrinsicMatrixFromMLIntrinsics(camera_intrinsics);

    return MLResult_Ok;
  }

  MLResult StopCapture() {
    MLCameraCaptureVideoStop(recorder_camera_context_);
    return MLResult_Ok;
  }

  MLResult DestroyCamera()
  {
    if (MLHandleIsValid(recorder_camera_context_))
    {
      MLCameraDisconnect(recorder_camera_context_);
      recorder_camera_context_ = ML_INVALID_HANDLE;
      recorder_camera_device_available_ = false;
    }
    MLCameraDeInit();

    MLCVCameraTrackingDestroy(cv_camera_tracker_);
    MLHeadTrackingDestroy(head_tracker_);

    return MLResult_Ok;
  }

  MLResult SetupCamera()
  {
    MLResult res;
    if (MLHandleIsValid(recorder_camera_context_))
    {
      return MLResult_Ok;
    }
    MLCameraDeviceAvailabilityStatusCallbacks device_availability_status_callbacks = {};
    MLCameraDeviceAvailabilityStatusCallbacksInit(&device_availability_status_callbacks);

    device_availability_status_callbacks.on_device_available = [](const MLCameraDeviceAvailabilityInfo *avail_info)
    {
      CheckDeviceAvailability(avail_info, true);
    };
    device_availability_status_callbacks.on_device_unavailable = [](const MLCameraDeviceAvailabilityInfo *avail_info)
    {
      CheckDeviceAvailability(avail_info, false);
    };

    res = MLCameraInit(&device_availability_status_callbacks, this);
    if (res != MLResult_Ok) { return res; }
    { // wait for maximum 2 seconds until camera becomes available
      std::unique_lock<std::mutex> lock(camera_device_available_lock_);
      camera_device_available_condition_.wait_for(lock, 2000ms, [&]() { return recorder_camera_device_available_; });
    }

    if (!recorder_camera_device_available_)
    {
      ALOGE("Timed out waiting for Main camera!");
      return MLResult_Timeout;
    }
    else
    {
      ALOGI("Main camera is available!");
    }

    MLCameraConnectContext camera_connect_context = {};
    MLCameraConnectContextInit(&camera_connect_context);
    camera_connect_context.cam_id = MLCameraIdentifier_MAIN;
    camera_connect_context.flags = MLCameraConnectFlag_CamOnly;
    camera_connect_context.enable_video_stab = false;
    res = MLCameraConnect(&camera_connect_context, &recorder_camera_context_);
    if (res != MLResult_Ok) { return res; }
    res = SetCameraRecorderCallbacks();
    if (res != MLResult_Ok) { return res; }

    res = MLCVCameraTrackingCreate(&cv_camera_tracker_);
    if (res != MLResult_Ok) { return res; }
    res = MLHeadTrackingCreate(&head_tracker_);
    if (res != MLResult_Ok) { return res; }

    return MLResult_Ok;
  }

  static void CheckDeviceAvailability(const MLCameraDeviceAvailabilityInfo *device_availability_info, bool is_available)
  {
    if (device_availability_info == nullptr)
    {
      return;
    }
    CameraForCV *this_app = static_cast<CameraForCV *>(device_availability_info->user_data);
    if (this_app && device_availability_info->cam_id == MLCameraIdentifier_MAIN)
    {
      this_app->recorder_camera_device_available_ = is_available;
      this_app->camera_device_available_condition_.notify_one();
    }
  }

  MLResult SetCameraRecorderCallbacks()
  {
    MLResult res;
    MLCameraDeviceStatusCallbacks camera_device_status_callbacks = {};
    MLCameraDeviceStatusCallbacksInit(&camera_device_status_callbacks);

    camera_device_status_callbacks.on_device_error = [](MLCameraError err, void *)
    {
      ALOGE("on_device_error(%s) callback called for recorder camera", EnumHelpers::GetMLCameraErrorString(err));
    };

    camera_device_status_callbacks.on_device_disconnected = [](MLCameraDisconnectReason reason, void *)
    {
      ALOGE("on_device_disconnected(%s) callback called for recorder camera",
            EnumHelpers::GetMLCameraDisconnectReasonString(reason));
    };
    
    res = MLCameraSetDeviceStatusCallbacks(recorder_camera_context_, &camera_device_status_callbacks, this);
    if (res != MLResult_Ok) { return res; }

    MLCameraCaptureCallbacks camera_capture_callbacks = {};
    MLCameraCaptureCallbacksInit(&camera_capture_callbacks);

    camera_capture_callbacks.on_capture_failed = [](const MLCameraResultExtras *, void *)
    {
      ALOGI("on_capture_failed callback called for recorder camera");
    };

    camera_capture_callbacks.on_capture_aborted = [](void *)
    {
      ALOGI("on_capture_aborted callback called for recorder camera");
    };

    camera_capture_callbacks.on_video_buffer_available = OnVideoAvailable;
    res = MLCameraSetCaptureCallbacks(recorder_camera_context_, &camera_capture_callbacks, this);
    if (res != MLResult_Ok) { return res; }
    return MLResult_Ok;
  }

  std::string GetTimeStr() const
  {
    const std::time_t tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    const std::tm tm = *std::localtime(&tt);
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return ss.str();
  }

  MLResult SetupCaptureSize()
  {
    int32_t width = 0, height = 0;
    uint32_t streams_max = 0;
    MLCameraGetNumSupportedStreams(recorder_camera_context_, &streams_max);

    typedef struct StreamCapsInfo
    {
      uint32_t stream_caps_max;
      MLCameraCaptureStreamCaps *stream_caps;
    } StreamCapsInfo;

    StreamCapsInfo *stream_caps_info = nullptr;
    stream_caps_info = (StreamCapsInfo *)malloc(streams_max * sizeof(StreamCapsInfo));
    if (stream_caps_info == nullptr)
    {
      ALOGE("Memory Allocation for StreamCapsInfo failed");
      return MLResult_UnspecifiedFailure;
    }

    for (uint32_t i = 0; i < streams_max; i++)
    {
      stream_caps_info[i].stream_caps_max = 0;
      stream_caps_info[i].stream_caps = nullptr;
      MLCameraGetStreamCaps(recorder_camera_context_, i, &stream_caps_info[i].stream_caps_max, nullptr);
      stream_caps_info[i].stream_caps =
          (MLCameraCaptureStreamCaps *)malloc(stream_caps_info[i].stream_caps_max * sizeof(MLCameraCaptureStreamCaps));
      MLCameraGetStreamCaps(recorder_camera_context_, i, &stream_caps_info[i].stream_caps_max, stream_caps_info[i].stream_caps);

      for (uint32_t j = 0; j < stream_caps_info[i].stream_caps_max; j++)
      {
        const MLCameraCaptureStreamCaps capture_stream_caps = stream_caps_info[i].stream_caps[j];
        if (capture_stream_caps.capture_type == MLCameraCaptureType_Video)
        {
          ALOGD("Stream %d, Width %d, Height %d", j, capture_stream_caps.width, capture_stream_caps.height);
          if (capture_stream_caps.height == 960)
          {
            width = capture_stream_caps.width;
            height = capture_stream_caps.height;
            break;
          }
        }
      }
    }

    for (uint32_t i = 0; i < streams_max; i++)
    {
      if (stream_caps_info[i].stream_caps != nullptr)
      {
        free(stream_caps_info[i].stream_caps);
      }
    }
    free(stream_caps_info);

    if (width > 0 && height > 0)
    {
      capture_width_ = width;
      capture_height_ = height;
      if ((capture_width_ * capture_height_) > (2048 * 1536))
      {
        capture_fps_ = MLCameraCaptureFrameRate_30FPS;
      }
      else
      {
        ALOGD("Setting capture fps to 60");
        capture_fps_ = MLCameraCaptureFrameRate_60FPS;
      }
    }

    return MLResult_Ok;
  }


  MLTransform current_transform;
  MLHandle cv_camera_tracker_;
  MLHandle head_tracker_;
  const MLCameraOutput *current_output;
  bool recorder_camera_device_available_, has_recording_started_;
  std::mutex camera_device_available_lock_;
  std::condition_variable camera_device_available_condition_;
  int32_t capture_width_, capture_height_;
  std::vector<uint8_t> framebuffer_;
  MLCameraContext recorder_camera_context_;
  torch::Tensor intrinsic_inverse_matrix;
  std::string current_filename_, current_filename_photo_;
  uint64_t current_capture_len_ms_;
  MLCameraCaptureFrameRate capture_fps_;
};
