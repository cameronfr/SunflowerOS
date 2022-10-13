#include <torch/script.h>
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "debug.h"

#include "OneEuroFilter.cpp"

#include "../../StereoKit/StereoKitC/libraries/sokol_time.h"

#include <stereokit.h>
#include <android/log.h>
#include <iostream>
#define LOG_TAG "SunflowerOS"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

using namespace sk;
using namespace torch::indexing;

// template <typename T> const char* tPrint(T t) {
//   std::ostringstream os;
//   os << t << std::endl;
//   printf("Tensor: %s", os.str().c_str());
// }

class TfLiteInterpreterWrapper {
public:
  TfLiteInterpreterWrapper(char *modelPath) {
    LOGD("TfLiteInterpreterWrapper constructor with arg\n");
    model = TfLiteModelCreateFromFile(modelPath);
    options = TfLiteInterpreterOptionsCreate();
    // Enable gpu delegate
    gpu_options = TfLiteGpuDelegateOptionsV2Default();
    gpu_options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
    gpu_options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
    gpu_options.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
    gpu_options.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
    gpu_options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
    delegate = TfLiteGpuDelegateV2Create(&gpu_options);
    TfLiteInterpreterOptionsAddDelegate(options, delegate);
    interpreter = TfLiteInterpreterCreate(model, options);
    TfLiteInterpreterAllocateTensors(interpreter);
  }

  ~TfLiteInterpreterWrapper() {
    LOGD("TfLiteInterpreterWrapper destructor\n");
    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);
    TfLiteGpuDelegateV2Delete(delegate);
  }
  TfLiteModel *model;
  TfLiteInterpreterOptions *options;
  TfLiteGpuDelegateOptionsV2 gpu_options;
  TfLiteDelegate *delegate;
  TfLiteInterpreter *interpreter;
};

class PoseModel {
public:

  PoseModel() : 
    pose_detector_tflite("/data/data/com.termux/files/home/MagicLeap2-Synced/models/pose_detection.tflite"),
    pose_landmarks_tflite("/data/data/com.termux/files/home/MagicLeap2-Synced/models/pose_landmark_lite.tflite") {
      main_keypoints_filter.Initialize(0, torch::zeros({33, 2}), torch::zeros({33, 2}), 0.005, 0.03875, 1.0);
      aux_keypoints_filter.Initialize(0, torch::zeros({2, 2}), torch::zeros({2, 2}), 0.008, 0.01, 1.0);
      pose_detection_anchors = GenerateAnchors();
      LOGD("Person Detec Anchors Count : %d\n", pose_detection_anchors.size(0));
      latestPose = torch::zeros({33, 3}); //x,y,z in image coordinates;
      // TfLiteInterpreter* pose_tflite_interpreter = CreateTfLiteInterpreter("/data/data/com.termux/files/home/MagicLeap2-Synced/models/pose_landmark_lite.tflite");
  }
  ~PoseModel() {
  };

  void debugShowImg (torch::Tensor inImg, tex_t debugTex) {
    torch::Tensor debugTensor = torch::ones({inImg.size(0)+8, inImg.size(1)+8, 4}, torch::kByte) * 255;
    debugTensor.index_put_({Slice(4, 4+inImg.size(0)),
                            Slice(4, 4+inImg.size(1)),
                            Slice()},
                            inImg);
    tex_set_colors(debugTex, debugTensor.size(1), debugTensor.size(0), (void*)debugTensor.data_ptr());
  };

  void  addRedDot (torch::Tensor t, int p0, int p1) {
    int sz = 12;
    int minP0 = std::max(0, p0-sz);
    int maxP0 = std::min((int)t.size(0), p0+sz);
    int minP1 = std::max(0, p1-sz);
    int maxP1 = std::min((int)t.size(1), p1+sz);
    uint8_t red[] = {255, 0, 0, 255};
    torch::Tensor redT = torch::from_blob(red, {4}, torch::kByte);
    t.index_put_({Slice(minP0, maxP0), Slice(minP1, maxP1), Slice()}, redT);
  };

  // std::tuple<torch::Tensor, std::function<torch::Tensor(torch::Tensor, bool)>> MakeImageInputTensor(torch::Tensor img, torch::Tensor box, float rotation) {
  //   // Input: raw image in RGB 0-255, HxWXC format
  //   // box is [yCenter, xCenter, ySize, xSize] in coords of raw Image




  // }


  std::tuple<torch::Tensor, std::function<torch::Tensor(torch::Tensor, bool)>> ResizeImage(torch::Tensor image, int target_size, bool resize_longer_side) {
    // Expects HxWXC image tensor, kByte, 0-255. Returns in same format.
    if (resize_longer_side) {
      int targetd0, targetd1;
      float scale;
      if (image.size(0) > image.size(1)) {
        targetd0 = target_size;
        targetd1 = (int)(image.size(1) * target_size / image.size(0));
        scale = (float)target_size / image.size(0);
      } else {
        targetd0 = (int)(image.size(0) * target_size / image.size(1));
        targetd1 = target_size;
        scale = (float)target_size / image.size(1);
      }

      torch::Tensor inputToResizeFn = image.permute({2, 0, 1}).unsqueeze_(0).to(torch::kFloat);
      torch::Tensor resizedRaw = torch::upsample_bilinear2d(inputToResizeFn, {targetd0, targetd1}, false);
      torch::Tensor resized = resizedRaw.squeeze_(0).permute({1, 2, 0}).to(torch::kByte);

      torch::Tensor padded = 0*torch::ones({target_size, target_size, 4}, torch::kByte) ;
      int d0 = (padded.size(0) - resized.size(0)) / 2;
      int d1 = (padded.size(1) - resized.size(1)) / 2;
      padded.index_put_({Slice(d0, d0 + resized.size(0)),
                        Slice(d1, d1 + resized.size(1)),
                        Slice()},
                        resized);
      
      // Map from coords on resized to coords on original image
      auto coordMapFn = [d0, d1, scale](torch::Tensor coords, bool onlyScale) {
        if (onlyScale) {
          return (1.0f / scale) * coords;
        } else {
          torch::Tensor addedPadding = torch::ones({2}, torch::kLong); addedPadding[0] = d0; addedPadding[1] = d1;
          return (1.0 / scale) * (coords - addedPadding); 
        }
      };

      return std::make_tuple(padded, coordMapFn);
    } else {
      LOGD("UNIMPLEMENTED\n");
      // throw exception
      throw std::runtime_error("UNIMPLEMENTED");
    }
  }

  virtual torch::Tensor GetLatestPose() {
    return latestPose; // 33x(x,y,z) in image coordinates
  }

  virtual torch::Tensor GetLatestPoseVisibilities() {
    return latestPoseVisibilities; // 33
  }

  // takes rawImage. Returns 4 tensor of corners of bounding box, in original img space, (mind0, maxd0, mind1, maxd1). Second return bool is whether it found a person. 
  virtual std::tuple<torch::Tensor, bool> PersonDetectorROI(torch::Tensor rawImage, tex_t debugTex) {
    std::tuple resizeOut = ResizeImage(rawImage, 224, true);
    torch::Tensor resizeOutTensor = std::get<0>(resizeOut);
    std::function<torch::Tensor(torch::Tensor, bool)> resizeOutCoordMap = std::get<1>(resizeOut);

    TfLiteTensor* inputTensorTf = TfLiteInterpreterGetInputTensor(pose_detector_tflite.interpreter, 0);
    torch::Tensor inputTensor = resizeOutTensor.index({"...", Slice(0, 3)}).to(torch::kFloat32).div_(255.0f).sub_(0.5f).div_(0.5f); 
    int status = TfLiteTensorCopyFromBuffer(inputTensorTf, inputTensor.data_ptr(), inputTensor.nbytes());
    if (status != kTfLiteOk) {
      LOGD("Error copying from buffer");
      throw std::runtime_error("Error copying from buffer");
    }

    status = TfLiteInterpreterInvoke(pose_detector_tflite.interpreter);
    if (status != kTfLiteOk) {
      LOGD("Error invoking %d", status);
      throw std::runtime_error("Error invoking");
    }

    // // // Extract the output tensor data.
    torch::Tensor outputKeypoints = torch::from_blob(TfLiteInterpreterGetOutputTensor(pose_detector_tflite.interpreter, 0)->data.data, {1, 2254, 12}, torch::kFloat32).squeeze();
    torch::Tensor outputConfidence = torch::from_blob(TfLiteInterpreterGetOutputTensor(pose_detector_tflite.interpreter, 1)->data.data, {1, 2254, 1}, torch::kFloat32).squeeze();
    // // NOTE: if input is d0, d1 (HxW) then output coords are d1, d0 -- i.e. x, y, and anchors expect that too
    
    torch::Tensor outputProbabilities = (1 / (1 + torch::exp(-outputConfidence)));

    // // Get top 1 detections as a mask, filter those that are lower than 0.5 confidence. If want actual top 4, need to do overlap based filtering.
    std::tuple topk = torch::topk(outputProbabilities, 1, 0, true, true); //Tuple with (values, indices)
    torch::Tensor candidateMask = std::get<1>(topk).index({std::get<0>(topk) > 0.5f});
    torch::Tensor candidateKeypoints = outputKeypoints.index({candidateMask, Slice()});

    // Single detection (or 0 detection). x_center,y_center, w, h, x1, y1, x2, y2, x3, y3, x4, y4.
    torch::Tensor detectionData = candidateKeypoints.reshape({-1, 6, 2});
    torch::Tensor detectionKpts = detectionData.index({Slice(), Slice(2, 6), Slice()});
    // torch::Tensor detectionBoxSize = detectionData.index({Slice(), Slice(1), Slice()}); //ignore w, h cause it doesn't seem to mean anything or seems to be face bounding box.

    if (detectionKpts.size(0) > 0) {
      detectionKpts = (detectionKpts / 224.0f) * pose_detection_anchors.index({candidateMask, None, Slice(2, 4)}) + pose_detection_anchors.index({candidateMask, None, Slice(0, 2)});
      detectionKpts = detectionKpts * 224.0f;
      detectionKpts = torch::flip(detectionKpts, {-1}); // flip x and y so that y is first,
      detectionKpts = resizeOutCoordMap(detectionKpts, false); // put into original image coordinates
      torch::Tensor allKpts = detectionKpts.reshape({-1, 2}); //1, 4, 2 -> 4, 2

      // Get box around person
      torch::Tensor chestToHeadVector = allKpts[1] - allKpts[0]; // 1 is head lips-ish, 0 is chest, 3 is head nose-ish
      torch::Tensor hips = allKpts[0] - chestToHeadVector * 1.5;
      // printf("hips: %f, %f", hips[0].item<float>(), hips[1].item<float>());
      // addRedDot(rawImage, allKpts[3][0].item<int>(), allKpts[3][1].item<int>()); // 1 is head center
      // addRedDot(rawImage, allKpts[1][0].item<int>(), allKpts[1][1].item<int>()); // 3 is a
      // debugShowImg(rawImage, debugTex);

      int radius = (int)2 * torch::square(allKpts[1] - hips).sum().sqrt().item<float>(); 
      // int mind0 = std::max(0, hips[0].item<int>() - radius);
      // int maxd0 = std::min((int)rawImage.size(1), hips[0].item<int>() + radius);
      // int mind1 = std::max(0, hips[1].item<int>() - radius);
      // int maxd1 = std::min((int)rawImage.size(0), hips[1].item<int>() + radius);
      int mind0 = hips[0].item<int>() - radius;
      int maxd0 = hips[0].item<int>() + radius;
      int mind1 = hips[1].item<int>() - radius;
      int maxd1 = hips[1].item<int>() + radius;

      int coords[] = {mind0, maxd0, mind1, maxd1};
      return std::make_tuple(torch::from_blob(coords, {4}, torch::kInt32).clone(), true);
    }
    return std::make_tuple(torch::zeros({4, 4}), false);
    

  }

  virtual bool ProcessImage(torch::Tensor rawImage, tex_t debugTex) {
    torch::Tensor roiBounds; // {4}
    bool haveROI;
    if (useLastLandmarkROI) {
      // std::cout << "Using ROI from landmarks" << std::endl;
      roiBounds = lastLandmarkROI;
      haveROI = true;
    } else {
      // LOGD("Using detector to find ROI");
      auto detectorResult = PersonDetectorROI(rawImage, debugTex);
      torch::Tensor detectorPersonBounds = std::get<0>(detectorResult);
      bool detectorFoundPerson = std::get<1>(detectorResult);

      roiBounds = detectorPersonBounds;
      haveROI = detectorFoundPerson;
    }

    if (haveROI) {
      int64_t prfCropImgStart = stm_now();
      // TODO: rotate ROI so that person is upright. 
      // Also, don't do transform until at end, so only copy tensor once
      int mind0 = roiBounds[0].item<int>();
      int maxd0 = roiBounds[1].item<int>();
      int mind1 = roiBounds[2].item<int>();
      int maxd1 = roiBounds[3].item<int>();

      if (mind0 >= maxd0 - 2 || mind1 >= maxd1 - 2) {
        std::cout << "ROI has width or height <= 2, returning" << std::endl;
        useLastLandmarkROI = false;
        return false;
      }

      // rawImage.index_put_({Slice(mind0, maxd0), Slice(mind1, maxd1), Slice()}, 
      //   rawImage.index({Slice(mind0, maxd0), Slice(mind1, maxd1), Slice()}) / 2
      // );
      // std::cout << "ROI: " << mind0 << " " << maxd0 << " " << mind1 << " " << maxd1 << std::endl;
      // torch::Tensor box = rawImage.index({Slice(mind0, maxd0), Slice(mind1, maxd1), Slice()});

      // Allow roi to be out of bounds of original image (i.e. negative or > img size)
      int mind0Clamped = std::max(0, mind0);
      int maxd0Clamped = std::min((int)rawImage.size(0), maxd0);
      int mind1Clamped = std::max(0, mind1);
      int maxd1Clamped = std::min((int)rawImage.size(1), maxd1);
      torch::Tensor box = torch::zeros({maxd0 - mind0, maxd1 - mind1, 4});
      box.index_put_({Slice(mind0Clamped - mind0, maxd0Clamped - mind0), Slice(mind1Clamped - mind1, maxd1Clamped - mind1), Slice()}, 
        rawImage.index({Slice(mind0Clamped, maxd0Clamped), Slice(mind1Clamped, maxd1Clamped), Slice()})
      );

      auto boxResizeOut = ResizeImage(box, 256, true);
      torch::Tensor boxResizeOutTensor = std::get<0>(boxResizeOut);
      std::function<torch::Tensor(torch::Tensor, bool)> boxResizeOutCoordMap = std::get<1>(boxResizeOut);

      int64_t prfCropImageDur = stm_since(prfCropImgStart);
      // LOGD("CropImage took %fms", stm_ms(prfCropImageDur));
      int64_t prfModelStart = stm_now();

      // Run box through landmark estimator
      TfLiteTensor* inputTensorTf2 = TfLiteInterpreterGetInputTensor(pose_landmarks_tflite.interpreter, 0);
      torch::Tensor inputTensor2 = boxResizeOutTensor.index({"...", Slice(0, 3)}).to(torch::kFloat32).div_(255.0f); 
      // std::cout << "inputTensor2: " << inputTensor2.sizes() << std::endl;
      int status = TfLiteTensorCopyFromBuffer(inputTensorTf2, inputTensor2.data_ptr(), inputTensor2.nbytes());
      if (status != kTfLiteOk) {
        LOGD("Error copying from buffer");
        throw std::runtime_error("Error copying from buffer");
      }

      TfLiteInterpreterInvoke(pose_landmarks_tflite.interpreter);

      const TfLiteTensor* tfLandmarkOutput = TfLiteInterpreterGetOutputTensor(pose_landmarks_tflite.interpreter, 0);
      torch::Tensor landmarkOutputRaw = torch::from_blob(tfLandmarkOutput->data.data, {35, 5}, torch::kFloat32); //actually 35, last two are auxillary
      
      int64_t prfModelDur = stm_since(prfModelStart);
      // LOGD("Model took %fms", stm_ms(prfModelDur));

      long long time_us = stm_us(stm_now());

      torch::Tensor landmarkOutputFiltered = landmarkOutputRaw;

      // landmarkOutputMain is 33 keypoints, (y,x) 
      torch::Tensor landmarkOutputYX = torch::flip(landmarkOutputFiltered.index({Slice(), Slice(0, 2)}), {-1});
      for (int i = 33; i < landmarkOutputYX.size(0); i++) {
        addRedDot(boxResizeOutTensor, landmarkOutputYX[i][0].item<int>(),  landmarkOutputYX[i][1].item<int>());
      }
      debugShowImg(boxResizeOutTensor, debugTex);
      torch::Tensor landmarkOutputZ = landmarkOutputFiltered.index({Slice(), Slice(2, 3)});
      int boxStartOffsetArr[] = {mind0, mind1};
      torch::Tensor boxStartOffset = torch::from_blob(boxStartOffsetArr, {2}, torch::kInt32).clone();
      landmarkOutputYX = boxResizeOutCoordMap(landmarkOutputYX, false) + boxStartOffset;
      // landmarkOutputZ = boxResizeOutCoordMap(landmarkOutputZ, true); //try and keep Z in same scale as YX, even though it's not really a coordinate
      landmarkOutputZ = (rawImage.size(0) / 256.0f) * landmarkOutputZ; // this seems more accurate as z-scale

      // TODO: scale of keypoints changes here. Normalize by box size or smthn before put in? If filter aux keypoints before the boxResizeOutCoordMap, get weird bouncy feedback.
      landmarkOutputYX.index_put_({Slice(33, 35), Slice()}, 
        aux_keypoints_filter.filter(time_us, landmarkOutputYX.index({Slice(33, 35), Slice()}))
      );

      // for (int i = 34; i < landmarkOutputYX.size(0); i++) {
      //   addRedDot(rawImage, landmarkOutputYX[i][0].item<int>(),  landmarkOutputYX[i][1].item<int>());
      // }
      // debugShowImg(rawImage, debugTex);

      torch::Tensor visibilityProbs = 1 / (1 + torch::exp(-landmarkOutputRaw.index({Slice(), 3})));

      // float maxVisibilityProb = (visibilityProbs).mean().item<float>();
      // get median visibility prob using torch sorting
      torch::Tensor visibilityProbsSorted = std::get<0>(visibilityProbs.sort(0));
      float maxVisibilityProb = visibilityProbsSorted[visibilityProbsSorted.size(0) / 2].item<float>();

      latestPose = torch::concat({torch::flip(landmarkOutputYX, {-1}), landmarkOutputZ}, 1); // 33x [x_img, y_img, z_img_ish]
      latestPoseVisibilities = visibilityProbs; 

      // Save ROI for next frame, if the track is good
      if (maxVisibilityProb > 0.5) {
        // Works well to use mediapipe method w/ mind0 etc allowed to be negative. s.t. center aux keypoint is always at center of imag fed to mediapipe.
        // get two aux keypoints 
        torch::Tensor centerKpt = landmarkOutputYX.index({33});
        torch::Tensor scaleKpt = landmarkOutputYX.index({34});
        int radius = 1.25*(centerKpt - scaleKpt).norm().item<int>(); //i.e., width is 1.25r+1.25r = 2r*1.25 -- what mediapipe is doing
        int roi_mind0 = centerKpt[0].item<int>() - radius;
        int roi_maxd0 = centerKpt[0].item<int>() + radius;
        int roi_mind1 = centerKpt[1].item<int>() - radius;
        int roi_maxd1 = centerKpt[1].item<int>() + radius;

        torch::Tensor roiBoundsLandmarks = torch::from_blob((int[]){roi_mind0, roi_maxd0, roi_mind1, roi_maxd1}, {4}, torch::kInt32).clone();

        lastLandmarkROI = roiBoundsLandmarks;
        useLastLandmarkROI = true;
        return true;
      } else {
        LOGD("maxVisibilityProb too low, not using this as ROI, val is %f", maxVisibilityProb);
        // Reset ROI-box filter and keypoints filters.
        aux_keypoints_filter.ResetHistory();
        main_keypoints_filter.ResetHistory();
        useLastLandmarkROI = false;
        return false; //return false to make sure 
      }
      // return true;
    } else {
      // std::cout << "DON'T HAVE AN ROI" << std::endl;
      return false;
    }
  }
public:
  OneEuroFilter main_keypoints_filter;
  OneEuroFilter aux_keypoints_filter;
private: 
  // Anchors for mediapipe pose_detection (not pose landmarks)
  // They should put this in the model tflite....
  // Output n_anchors x 4 (x, y, w, h)
  torch::Tensor GenerateAnchors() {
    /*
        num_layers: 5
        min_scale: 0.1484375
        max_scale: 0.75
        input_size_height: 224
        input_size_width: 224
        anchor_offset_x: 0.5
        anchor_offset_y: 0.5
        strides: 8
        strides: 16
        strides: 32
        strides: 32
        strides: 32
        aspect_ratios: 1.0
        fixed_anchor_size: true
      */
    // Input
    int num_layers = 5;
    float min_scale = 0.1484375;
    float max_scale = 0.75;
    int input_size_height = 224;
    int input_size_width = 224;
    float anchor_offset_x = 0.5;
    float anchor_offset_y = 0.5;
    std::vector<int> strides = {8, 16, 32, 32, 32};
    std::vector<float> aspect_ratios_option = {1.0};
    bool fixed_anchor_size = true;
    float interpolated_scale_aspect_ratio = 1.0f;
    // Output
    std::vector<torch::Tensor> anchors_list;

    auto CalculateScale = [](float min_scale, float max_scale, int stride_index, int num_strides) -> float {
      if (num_strides == 1) {
        return (min_scale + max_scale) * 0.5f;
      } else {
        return min_scale + (max_scale - min_scale) * 1.0 * stride_index / (num_strides - 1.0f);
      }
    };

    int layer_id = 0;
    while (layer_id < num_layers) {
      std::vector<float> anchor_height;
      std::vector<float> anchor_width;
      std::vector<float> aspect_ratios;
      std::vector<float> scales;

      // For same strides, we merge the anchors in the same order.
      int last_same_stride_layer = layer_id;
      while (last_same_stride_layer < strides.size() && strides[last_same_stride_layer] == strides[layer_id]) {
        const float scale = CalculateScale(min_scale, max_scale, last_same_stride_layer, strides.size());
        for (int aspect_ratio_id = 0; aspect_ratio_id < aspect_ratios_option.size(); ++aspect_ratio_id) {
          aspect_ratios.push_back(aspect_ratios_option[aspect_ratio_id]);
          scales.push_back(scale);
        }
        if (interpolated_scale_aspect_ratio > 0.0) {
          const float scale_next =
              last_same_stride_layer == strides.size() - 1
                  ? 1.0f
                  : CalculateScale(min_scale, max_scale,
                                    last_same_stride_layer + 1,
                                    strides.size());
          scales.push_back(std::sqrt(scale * scale_next));
          aspect_ratios.push_back(interpolated_scale_aspect_ratio);
        }
        last_same_stride_layer++;
      }

      for (int i = 0; i < aspect_ratios.size(); ++i) {
        const float ratio_sqrts = std::sqrt(aspect_ratios[i]);
        anchor_height.push_back(scales[i] / ratio_sqrts);
        anchor_width.push_back(scales[i] * ratio_sqrts);
      }

      int feature_map_height = 0;
      int feature_map_width = 0;
      const int stride = strides[layer_id];
      feature_map_height = std::ceil(1.0f * input_size_height / stride);
      feature_map_width = std::ceil(1.0f * input_size_width / stride);

      for (int y = 0; y < feature_map_height; ++y) {
        for (int x = 0; x < feature_map_width; ++x) {
          for (int anchor_id = 0; anchor_id < anchor_height.size(); ++anchor_id) {
            // TODO: Support specifying anchor_offset_x, anchor_offset_y.
            const float x_center =
                (x + anchor_offset_x) * 1.0f / feature_map_width;
            const float y_center =
                (y + anchor_offset_y) * 1.0f / feature_map_height;

            // Anchor new_anchor;
            torch::Tensor new_anchor = torch::zeros({4}, torch::kFloat32);
            new_anchor[0] = x_center;
            new_anchor[1] = y_center;

            if (fixed_anchor_size) {
              new_anchor[2] = anchor_width[anchor_id];
              new_anchor[3] = anchor_height[anchor_id];
            }

            anchors_list.push_back(new_anchor);
          }
        }
      }
      layer_id = last_same_stride_layer;
    }

    torch::Tensor anchors = torch::stack(anchors_list);

    return anchors;
  }

  TfLiteInterpreterWrapper pose_detector_tflite;
  TfLiteInterpreterWrapper pose_landmarks_tflite;
  torch::Tensor pose_detection_anchors;
  torch::Tensor lastLandmarkROI;
  torch::Tensor latestPose;
  torch::Tensor latestPoseVisibilities;
  bool useLastLandmarkROI = false;
};

extern "C" PoseModel* create() {
  return new PoseModel();
}
extern "C" void destroy(PoseModel* p) {
  delete p;
}

// Torch 16x16x4 byte tensor with 4 sections, one red, one green, one blue, one yellow
// torch::Tensor shrunk = torch::ones({16, 16, 4}, torch::kByte);
// uint8_t redArr[] = {255, 0, 0, 255};
// torch::Tensor redColor = torch::from_blob((void*)redArr, {1, 1, 4}, torch::kByte);
// uint8_t greenArr[] = {0, 255, 0, 255};
// torch::Tensor greenColor = torch::from_blob((void*)greenArr, {1, 1, 4}, torch::kByte);
// uint8_t blueArr[] = {0, 0, 255, 255};
// torch::Tensor blueColor = torch::from_blob((void*)blueArr, {1, 1, 4}, torch::kByte);
// uint8_t yellowArr[] = {255, 255, 0, 255};
// torch::Tensor yellowColor = torch::from_blob((void*)yellowArr, {1, 1, 4}, torch::kByte);

// shrunk.index_put_({Slice(0, 8, 1), Slice(0, 8, 1), Slice(0, 4, 1)}, redColor);
// shrunk.index_put_({Slice(0, 8, 1), Slice(8, 16, 1), Slice(0, 4, 1)}, greenColor);
// shrunk.index_put_({Slice(8, 16, 1), Slice(0, 8, 1), Slice(0, 4, 1)}, blueColor);
// shrunk.index_put_({Slice(8, 16, 1), Slice(8, 16, 1), Slice(0, 4, 1)}, yellowColor);

// shrunk = shrunk.index({
//   Slice(0, 16, 2),
//   Slice(0, 16, 1),
//   Slice()});