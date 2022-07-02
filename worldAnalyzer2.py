# Updated version. Simplified.

# pip install timm

import torch
import torchvision
import numpy as np
import sys
sys.path.insert(0, "/home/cameron/vitpose/") # load the modified mmpose from the vitpose repo. Modifiy mmpose/mmpose/__init__.py so that "1.5.0" max is replaced w/ current install
import mmdet.apis
import mmpose.apis
import os
import av
from PIL import Image

import plotly.express as px
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams["axes.grid"] = False
%config InlineBackend.figure_format='svg'

device = torch.cuda.current_device()

class DictToObject(object):
	def __init__(self, d):
		for a, b in d.items():
			if isinstance(b, (list, tuple)):
			   setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
			else:
			   setattr(self, a, obj(b) if isinstance(b, dict) else b)

# ------------------------------ HELPERS------------------------------

def keypointsCOCOToB36M(keypoints):
	keypoints_new = np.zeros((17, keypoints.shape[1]))
	# pelvis is in the middle of l_hip and r_hip
	keypoints_new[0] = (keypoints[11] + keypoints[12]) / 2
	# thorax is in the middle of l_shoulder and r_shoulder
	keypoints_new[8] = (keypoints[5] + keypoints[6]) / 2
	# head is in the middle of l_eye and r_eye
	keypoints_new[10] = (keypoints[1] + keypoints[2]) / 2
	# spine is in the middle of thorax and pelvis
	keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
	# rearrange other keypoints
	keypoints_new[[1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = \
		keypoints[[12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]]
	return keypoints_new

# in the mmdet Repo
def loadDetector():
	os.chdir("/home/cameron/mmdetection/")
	modelConfig = "configs/yolox/yolox_x_8x8_300e_coco.py"
	modelCheckpoint = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth"
	model = mmdet.apis.init_detector(modelConfig, modelCheckpoint, device) # this is a torch module.
	sum([x.numel() for x in model.parameters()])
	return model

# input is [N H W (R,G,B)]. uint8.
def processFramesDetector(frames):
	imageTransforms = torchvision.transforms.Compose([
		torchvision.transforms.Pad((0, 0, 0, 560), fill=114),
		torchvision.transforms.Resize((640, 640)),
	])
	processed = torch.tensor(frames)[:, :, :, (2, 1, 0)] # BGR -> RGB
	processed = processed.permute(0, 3, 1, 2) # (Batch, W, H, C) -> (Batch, C, W, H)
	processed.shape
	processed = imageTransforms(processed)
	return processed

# input is output of pipeline
def forwardDetector(input, model):
	with torch.no_grad():
		# img_meta = {"img_shape": (640, 640), "flip": False, "scale_factor":1}
		img_meta = {"flip": False, 'ori_shape': (720, 1280, 3), 'img_shape': (360, 640, 3), 'pad_shape': (640, 640, 3), 'scale_factor': np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)} #ori_shape or some keys like that used when rescaling bboxes to match orig image. Otherwise they refer to post-process img.
		input = processedFrames.type(torch.FloatTensor).to(device)
		img_metas = [img_meta]*input.shape[0]
		out = model.forward_test([input], [img_metas], rescale=True)
	return out

videoRaw = av.open("/home/cameron/shibuyaTrim1.mp4")
# videoRaw.seek(3*10**6)
videoStream = videoRaw.streams[0]
videoFrames = videoRaw.decode(videoRaw.streams[0])
videoArray = np.empty((videoStream.frames, videoStream.height, videoStream.width, 3), dtype=np.uint8) #585MB for 10 seconds of video
for idx, frame in enumerate(videoRaw.decode(videoRaw.streams[0])):
	videoArray[idx] = np.array(frame.to_image())
actualFramecount = int(videoStream.base_rate * videoStream.duration * videoStream.time_base)
videoArray = videoArray[:actualFramecount]

frames = videoArray[:]#[0:100]

detectorModule = loadDetector()
processedFrames = processFramesDetector(frames).repeat((1, 1, 1, 1))
detectorOut = forwardDetector(processedFrames, detectorModule)

# resVis = detectorModule.show_result(frames[0], detectorOut[0], score_thr=0.5, bbox_color=None, text_color=(200, 200, 200), mask_color=None)
# plt.figure(figsize = (10,10))
# plt.imshow(resVis)
#
# VitPose repo
def loadPose2D():
	os.chdir("/home/cameron/")
	modelConfig = "vitpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py"
	modelCheckpoint = "vitpose-l-multi-coco.pth" #download from onedrive
	model = mmpose.apis.init_pose_model(modelConfig, modelCheckpoint, device) # this is a torch module.
	sum([x.numel() for x in model.parameters()])
	return model

from mmpose.apis import inference_top_down_pose_model, vis_pose_result, get_track_id # its messy / not good structure, resist urge to clean up
import copy
import tqdm

poseModel = loadPose2D()
len(detectorOut)

posesByFrame = []
nextTrackId = 0
lastPose2dOutWithTrackId = []
for frameIdx, detectionsInFrame in tqdm.tqdm(enumerate(detectorOut)):
	people = detectionsInFrame[0]
	confidences = people[:, 4]
	peopleBboxes = [{"bbox": x} for x in people[confidences > 0.4]]
	pose2dOut, _ = inference_top_down_pose_model(poseModel, frames[frameIdx][:, :, ::-1], person_results=peopleBboxes, bbox_thr=0.0, format="xyxy")

	# pose2dOut is list of {"keypoints" ..., "bbox": ...}, and pose2dOutWithTrackId adds a "track id" to each dict.
	pose2dOutWithTrackId, nextTrackId = get_track_id(pose2dOut, copy.deepcopy(lastPose2dOutWithTrackId), nextTrackId, use_oks=True, tracking_thr=0.3) # code deletes objs in last pose...
	posesByFrame.append(pose2dOutWithTrackId)
	lastPose2dOutWithTrackId = pose2dOutWithTrackId

	# vis = vis_pose_result(poseModel, frames[frameIdx], pose2dOut, kpt_score_thr=0.0, radius=4, thickness=2)
	# plt.figure(figsize = (10,10))
	# plt.imshow(vis)
	# break

posesByTrackId = {}
for frameIdx, poses in enumerate(posesByFrame):
	for pose in poses:
		track_id = pose["track_id"]
		if track_id not in posesByTrackId:
			posesByTrackId[track_id] = []
		else:
			if posesByTrackId[track_id][-1]["frameIdx"] != frameIdx - 1:
				raise "track is non contiguous, not expected"
		pose["frameIdx"] = frameIdx
		posesByTrackId[track_id].append(pose)


[print("track len", len(poses), "track id", poses[0]["track_id"]) for poses in posesByTrackId.values()]

for pose in posesByTrackId[17][::10]:
	vis = vis_pose_result(poseModel, frames[pose["frameIdx"]], [pose], kpt_score_thr=0.0, radius=4, thickness=2)
	plt.figure(figsize = (5,5))
	plt.imshow(vis)
plot3dPose()

# 17 is good canditate to do first then
keypointsFormattedForTrack = np.empty((len(frames), 17 ,2), dtype=np.float32)
keypointsFormattedForTrack.shape
startFrameForTrack = posesByTrackId[17][0]["frameIdx"]
endFrameForTrack = posesByTrackId[17][-1]["frameIdx"]
for i in range(0, startFrameForTrack):
	keypointsCOCO = posesByTrackId[17][0]["keypoints"][:, :2] #temporal pad left
	keypointsFormattedForTrack[i] = keypointsCOCOToB36M(keypointsCOCO)
for i in range(startFrameForTrack, endFrameForTrack+1):
	keypointsCOCO = posesByTrackId[17][i-startFrameForTrack]["keypoints"][:, :2]
	keypointsFormattedForTrack[i] = keypointsCOCOToB36M(keypointsCOCO)
for i in range(endFrameForTrack+1, len(frames)):
	keypointsCOCO = posesByTrackId[17][-1]["keypoints"][:, :2] #temporal pad right
	keypointsFormattedForTrack[i] = keypointsCOCOToB36M(keypointsCOCO)
import pickle
keypointsFormattedForTrack
keypointsFormattedForTrack.astype(np.float32, copy=False)
os.chdir("/home/cameron/3DMPP_Modified/mupots/est_p2ds")
pickle.dump([keypointsFormattedForTrack, None, None, None], open("0_Custom_00_00.pkl", "wb"))

startFrameForTrack

# Keypoint transfer definition
os.chdir("/home/cameron/3DMPP_Modified/util")
import datautil
import keypointsTools
mpiiToH36m = lambda p: keypointsTools.convert_kps(p.transpose(-1, -2), "mpii3d_test", "h36m").transpose(-1,-2)
def mpiiToH36m(pts):
	perm = keypointsTools.get_perm_idxs("mpii3d_test", "h36m")
	return pts[..., perm, :]

# Stage 1 Intermediate results (pred, topdown_pts). pred_depth step is if want to use gt_depth, so skip.
predTopDown = pickle.load(open("/home/cameron/3DMPP_Modified/mupots/pred/1.pkl", "rb"))
predTopDown = pickle.load(open("/home/cameron/3D-Multi-Person-Pose/mupots/pred/1.pkl", "rb"))
plot3dPose(mpiiToH36m(predTopDown[10, 0, ...].transpose(1, 0)))

# Save frames to disk
os.chdir("/home/cameron/3DMPP_Modified")
# os.mkdir("images_store_tmp")
os.chdir("images_store_tmp")
for idx, img in enumerate(frames):
	Image.fromarray(img).save(str(idx) + ".png")

# Stage 2 Bottom up.
predBtmUp = pickle.load(open("/home/cameron/3DMPP_Modified/mupots/MUPOTS_Preds_btmup_transformed.pkl", "rb"))
predBtmUp = pickle.load(open("/home/cameron/3D-Multi-Person-Pose/mupots/MUPOTS_Preds_btmup_transformed.pkl", "rb"))
len(predBtmUp[1]) #frames
len(predBtmUp[1][0]) #each frame has 4 of something?
predBtmUp[1][0][0]# frame name
len(predBtmUp[1][0][1]) # people idx?
predBtmUp[1][0][1][0].shape #pose? (14,3)
predBtmUp[1][0][2] #depth?
predBtmUp[1][0][3][0].shape
len(predBtmUp[1][0][3]) #another people idx?
predBtmUp[1][10][3][0].shape #h36m ? pose (17,3) transformed
plot3dPose(mpiiToH36m(predBtmUp[1][10][3][0])) #mupots looks good, mine looks trash. note can barely seem the man in frame at this point.


# Stage 3 integration.
integratedDepth1 = pickle.load(open("/home/cameron/3DMPP_Modified/mupots/pred_dep_inte/00_00.pkl", "rb"))
# integratedDepth1 = pickle.load(open("/home/cameron/3DMPP_Modified/mupots/pred_dep_bu/00_00.pkl", "rb"))
# integratedDepth1 = pickle.load(open("/home/cameron/3DMPP_Modified/mupots/pred_dep/00_00.pkl", "rb"))
# integratedDepth2 = pickle.load(open("/home/cameron/3D-Multi-Person-Pose/mupots/pred_dep_inte/00_00.pkl", "rb"))
# integratedDepth2 = pickle.load(open("/home/cameron/3D-Multi-Person-Pose/mupots/pred_dep_bu/00_00.pkl", "rb"))
# integratedDepth2 = pickle.load(open("/home/cameron/3D-Multi-Person-Pose/mupots/pred_dep/00_00.pkl", "rb"))
integratedDepth1[0]/ integratedDepth1[-1] #depth prediction never changes?
# integratedDepth2[0]/ integratedDepth2[-1] #depth prediction never changes?

topdownDepth1 =pickle.load(open("/home/cameron/3DMPP_Modified/mupots/pred_dep/00_00.pkl", "rb"))
topdownDepth2 =pickle.load(open("/home/cameron/3D-Multi-Person-Pose/mupots/pred_dep/00_00.pkl", "rb"))
# Mismatch
gtDepth =pickle.load(open("/home/cameron/3D-Multi-Person-Pose/mupots/depths/00_00.pkl", "rb"))
gtDepth =pickle.load(open("/home/cameron/3D-Multi-Person-Pose/mupots/depths/00_01.pkl", "rb"))
gtDepth =pickle.load(open("/home/cameron/3D-Multi-Person-Pose/mupots/depths/05_00.pkl", "rb"))
plt.hist(gtDepth)
plt.hist(gtDepth)
plt.hist(gtDepth)
np.concatenate([gtDepth]).shape

topdownDepth1[::10]
integratedDepth1[::10]

os.chdir("/home/cameron/3DMPP_Modified")
import util.load_pred
integrated = pickle.load(open("/home/cameron/3DMPP_Modified/mupots/pred_inte/1.pkl", "rb"))
# integrated = pickle.load(open("/home/cameron/3D-Multi-Person-Pose/mupots/pred_inte/1.pkl", "rb"))

# depth doesn't work well, at least from these models. I think 3DPW-basd models, which don't use weak perspective, are way to go. Pose-lifter part of this prob useful, though. And can re-use models for training.
for frameNum in range(0, 110, 5):#[10, 50, 110]:
	# plot3dPose(mpiiToH36m(integrated[frameNum][0].transpose(1, 0)))
	integrated[frameNum][0].transpose(1,0)[0]
	integratedLoaded = util.load_pred.get_pred(0, frameNum)
	# plot3dPose(mpiiToH36m(integratedLoaded.squeeze().transpose(1, 0)))

	# Center at pelvis. Weak perspective, so just project (i.e. remove axis looking forward) and then scale according to depth.
	pose = mpiiToH36m(integratedLoaded.squeeze().transpose(1,0))
	pose = pose - pose[0]
	# plot3dPose(pose)
	poseSpineLength = np.linalg.norm(pose[7]-pose[0])

	# poseAdj3D = np.array([pose[:, 0], -pose[:, 1], pose[:, 2]]).transpose(1, 0) / (0.000005*integratedDepth1[frameNum])
	# poseAdj3D = poseAdj3D / poseSpineLength
	poseAdj3D = np.array([pose[:, 0], -pose[:, 1], pose[:, 2]]).transpose(1, 0) / (0.0015*integratedDepth1[frameNum])
	center2d = np.copy(keypointsFormattedForTrack[frameNum][0])
	center2d[1] = -center2d[1]
	poseAdj2D = poseAdj3D[:, [0, 1]] + center2d
	# plt.imshow(frames[frameNum])
	# plot2dPose(poseAdj2D, overlay=True)
	# plt.show()
	# plot2dPose(keypointsFormattedForTrack[frameNum])
	plot3dPose(pose)


# Try and figure out wtf the joint order is
annotMupots = datautil.load_annot("/home/cameron/3D-Multi-Person-Pose/MultiPersonTestSet/TS1/annot.mat")
#person, frame
indices = np.array([8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7]) - 1
points3Dgt = annotMupots[1][0]["annot3"][:, :].transpose(1, 0)
points2Dgt = annotMupots[1][0]["annot2"][:, :].transpose(1, 0)
plot2dPose(mpiiToH36m(points2Dgt))
annotMupots
points2Dgt
mpiiToH36m(points3Dgt)
plot3dPose(mpiiToH36m(points3Dgt))
import scipy.io as sio
# import importlib

# arr1pred_aligned = np.array([3799.8293, 3851.173 , 3913.9333, 3950.4255, 4006.3381, 4035.9968,
#        4055.0764, 4051.363 , 4010.5515, 3983.888 , 3965.2356, 3970.402 ,
#        3987.383 , 4007.0437, 4031.5925, 4052.4216, 4050.4739, 4101.534 ,
#        4159.7236, 4193.8877, 4213.802 , 4240.1113, 4277.1143, 4282.26  ,
#        4292.918 , 4329.1387, 4353.1123, 4386.756 , 4409.2334, 4421.409 ,
#        4408.6445, 4408.79  , 4407.507 , 4420.5195, 4386.574 , 4354.341 ,
#        4348.5557, 4397.842 , 4443.8545, 4461.081 , 4466.588 , 4467.457 ,
#        4490.623 , 4491.4697, 4511.712 , 4527.049 , 4540.6035, 4558.159 ,
#        4546.8994, 4544.05  , 4531.9736, 4546.9717, 4558.88  , 4570.6797,
#        4578.9365, 4582.2   , 4579.2607, 4578.5674, 4578.5605, 4565.713 ,
#        4557.133 , 4563.998 , 4584.837 , 4591.2334, 4590.9453, 4589.705 ,
#        4600.173 , 4607.793 , 4606.6494, 4592.4004, 4586.157 , 4594.1514,
#        4598.823 , 4596.339 , 4587.5225, 4577.288 , 4563.623 , 4557.6787,
#        4548.323 , 4542.4424, 4519.7275, 4493.8184, 4451.3955, 4424.412 ,
#        4401.409 , 4385.211 , 4358.3496, 4363.302 , 4334.6875, 4343.6436,
#        4292.2188, 4331.99  , 4342.291 , 4370.67  , 4360.7783, 4327.1025,
#        4300.664 , 4274.9316, 4284.8926, 4268.751 , 4255.627 , 4242.839 ,
#        4232.7764, 4246.8486, 4226.3857, 4223.8193, 4200.166 , 4185.7314,
#        4163.574 , 4150.332 , 4152.172 , 4149.33  , 4156.2773, 4159.3564,
#        4163.122 , 4162.0654, 4162.589 , 4159.716 , 4164.6777, 4145.8506,
#        4152.199 , 4155.5312, 4175.877 , 4128.209 , 4180.9014, 4099.0303,
#        4118.538 , 4098.0586, 4152.0273, 4198.491 , 4178.4453, 4111.4297,
#        4100.576 , 4113.0234, 4060.0237, 4036.8967, 3926.3054, 4016.2834,
#        3955.653 , 3978.0305, 3975.9202, 3957.4792, 3906.2766, 3800.9998,
#        3780.028 , 3774.9832, 3806.4128, 3772.7366, 3764.471 , 3751.8835,
#        3749.8367, 3737.5974, 3705.113 , 3685.26  , 3665.3826, 3642.7166,
#        3616.8147, 3592.1506, 3577.8508, 3586.0686, 3557.8674, 3524.5989,
#        3476.7693, 3457.9915, 3439.7952, 3443.2434, 3428.0598, 3415.7043,
#        3379.384 , 3356.0408, 3329.923 , 3330.8713, 3356.0002, 3355.536 ,
#        3348.7576, 3344.3508, 3366.1453, 3356.7239, 3351.2253, 3332.3772,
#        3349.72  , 3364.9514, 3366.8079, 3359.1638, 3337.9412, 3337.8503,
#        3333.1711, 3338.5247, 3324.7224, 3316.6458, 3317.3015, 3335.9607,
#        3351.6887, 3369.6008, 3377.4797, 3386.363 , 3386.3074, 3411.1633,
#        3434.4773, 3463.491 , 3468.9753, 2758.298 , 2764.1301, 2757.9685,
#        2752.612 , 2737.5793, 2732.342 , 2724.9246, 2724.2727, 2725.7786,
#        2722.3513, 2704.9915, 2692.2537, 2685.3728, 2699.2014, 2709.1555,
#        2710.8801, 2686.6287, 2686.3977, 2679.7356, 2695.2278, 2692.867 ,
#        2694.7126, 2697.7664, 2695.549 , 2712.5388, 2701.1848, 2688.1462,
#        2707.2483, 2712.1702, 2732.509 , 2715.8152, 2764.6487, 2774.6443,
#        2793.6492, 2761.5476, 2760.14  , 2749.9417, 2750.2688, 2738.8816,
#        2770.3455, 2786.5955, 2815.133 , 2831.2708, 2839.221 , 2847.1414,
#        2838.5183, 2860.3962, 2854.8567, 2846.2615, 2854.0706, 2843.7708,
#        2847.2317, 2832.7073, 2839.007 , 2846.5188, 2844.9465, 2832.6135,
#        2815.8464, 2807.7654, 2794.6506, 2811.4734, 2812.697 , 2844.5032,
#        2852.3777, 2880.2786, 2878.6301, 2916.596 , 2937.4548, 2950.9211,
#        2988.1907, 2986.1145, 3021.2366, 2990.203 , 3049.0037, 3042.9397,
#        3085.2883, 3111.24  , 3117.4368, 3144.2751, 3155.7854, 3197.8328,
#        3162.4958, 3169.8875, 3118.203 , 3112.6755, 3134.389 , 3151.9646,
#        3170.8914, 3138.7395, 3128.1794, 3114.924 , 3102.4172, 3054.0095,
#        3026.0833, 3010.3699, 3023.7546, 3030.0178, 3017.8562, 2975.3489,
#        2973.93  , 2914.261 , 2964.968 , 2956.071 , 2966.7673, 2928.5447,
#        2904.0852, 2911.0261, 2944.1628, 2945.9915, 2918.548 , 2890.2913,
#        2859.8494, 2850.1292, 2844.615 , 2867.2708, 2841.9934, 2821.3684,
#        2783.6184, 2740.8035, 2667.068 , 2679.8992, 2650.7507, 2676.4988,
#        2662.381 , 2661.5671, 2642.1765, 2627.2317, 2650.555 , 2678.1575,
#        2711.7288, 2717.1243, 2744.6697, 2760.6042, 2792.5618, 2823.1667,
#        2853.2512, 2856.2717, 2837.3943, 2813.8293, 2761.7874, 2737.5027,
#        2699.0095, 2675.6511, 2646.9138, 2637.8787, 2623.9158, 2635.2693,
#        2642.0056, 2647.5647, 2624.7512, 2666.7336, 2670.1638, 2668.1433,
#        2663.8645, 2694.2483, 2702.7996, 2683.327 , 2661.6692, 2676.4324,
#        2653.846 , 2637.3162, 2620.3489, 2635.0247, 2631.929 , 2610.5925,
#        2586.8342, 2574.2834, 2552.2537, 2521.1501, 2529.9856, 2512.992 ,
#        2537.4456, 2526.0208, 2545.636 , 2563.1946, 2561.8435, 2571.4275,
#        2519.5334, 2502.6682, 2469.0017, 2476.841 , 2495.0676, 2533.0173,
#        2572.7346, 2586.6335, 2600.9705, 2594.2947, 2586.448 , 2611.1492,
#        2630.406 , 2666.469 , 2645.7732, 2653.8074, 2645.5857, 2644.5793,
#        2612.8967, 2606.4163, 2607.464 , 2636.2463, 2656.1926, 2643.175 ],
#       dtype=np.float32)
# arr2gt = np.array([np.array([3660.54, 3687.5 , 3710.17, 3731.14, 3749.34, 3770.14, 3788.22,
#        3806.33, 3827.15, 3850.91, 3883.6 , 3916.61, 3966.71, 4015.84,
#        4071.82, 4106.35, 4147.6 , 4184.6 , 4210.22, 4237.54, 4261.92,
#        4280.4 , 4292.04, 4299.58, 4311.03, 4319.45, 4327.51, 4337.02,
#        4350.8 , 4361.14, 4381.31, 4396.29, 4414.94, 4431.71, 4447.21,
#        4463.06, 4489.21, 4513.8 , 4534.46, 4556.24, 4577.48, 4599.38,
#        4620.56, 4638.36, 4650.43, 4663.71, 4675.13, 4682.71, 4686.15,
#        4690.9 , 4693.97, 4698.02, 4701.17, 4703.61, 4704.  , 4710.2 ,
#        4709.8 , 4708.26, 4706.91, 4705.44, 4701.55, 4697.02, 4691.31,
#        4685.91, 4678.95, 4676.52, 4671.12, 4665.63, 4659.29, 4656.56,
#        4648.56, 4635.41, 4621.17, 4604.99, 4587.85, 4567.88, 4554.46,
#        4545.93, 4540.56, 4535.32, 4524.14, 4513.32, 4505.79, 4497.16,
#        4480.92, 4464.57, 4452.07, 4438.07, 4423.67, 4406.98, 4391.31,
#        4374.32, 4354.72, 4330.65, 4305.56, 4283.81, 4264.25, 4242.43,
#        4221.36, 4200.32, 4184.47, 4178.14, 4169.9 , 4161.09, 4148.31,
#        4137.19, 4127.34, 4123.49, 4121.55, 4122.64, 4122.95, 4123.94,
#        4126.25, 4135.24, 4141.9 , 4148.21, 4157.27, 4163.47, 4167.81,
#        4170.52, 4176.26, 4178.54, 4179.43, 4176.15, 4173.56, 4167.92,
#        4161.02, 4150.34, 4135.85, 4121.61, 4104.48, 4086.81, 4063.07,
#        4040.38, 4022.47, 3996.16, 3974.39, 3952.24, 3930.29, 3908.14,
#        3883.81, 3881.25, 3867.8 , 3856.29, 3836.61, 3822.89, 3811.2 ,
#        3795.39, 3778.54, 3757.14, 3736.33, 3713.85, 3689.59, 3661.42,
#        3630.34, 3591.73, 3557.78, 3523.98, 3497.82, 3473.59, 3451.56,
#        3430.35, 3410.15, 3394.92, 3377.41, 3365.78, 3351.46, 3342.89,
#        3330.78, 3319.93, 3310.44, 3299.26, 3293.29, 3289.37, 3284.91,
#        3280.56, 3280.33, 3279.75, 3278.84, 3284.88, 3289.07, 3294.44,
#        3296.55, 3297.62, 3300.14, 3305.06, 3309.84, 3317.06, 3315.36,
#        3320.28, 3331.08, 3337.48, 3348.59, 3360.26, 3373.22, 3380.5 ,
#        3397.3 , 3420.11, 3443.46, 3472.86, 3507.52], dtype=np.float32), np.array([2755.01, 2752.7 , 2751.3 , 2750.23, 2753.1 , 2754.56, 2751.27,
#        2749.58, 2749.16, 2746.17, 2741.75, 2738.57, 2735.47, 2737.58,
#        2753.68, 2764.85, 2766.44, 2769.57, 2777.33, 2784.61, 2780.1 ,
#        2766.87, 2759.95, 2748.25, 2738.69, 2732.5 , 2729.46, 2728.43,
#        2727.3 , 2731.24, 2740.68, 2754.94, 2768.74, 2779.01, 2783.86,
#        2785.6 , 2781.6 , 2784.78, 2787.75, 2790.55, 2786.67, 2780.4 ,
#        2771.9 , 2763.95, 2756.67, 2750.33, 2746.71, 2741.69, 2740.52,
#        2739.25, 2747.83, 2765.69, 2782.71, 2783.35, 2766.97, 2761.58,
#        2757.44, 2761.17, 2760.96, 2763.52, 2766.45, 2772.71, 2780.01,
#        2788.35, 2798.84, 2809.69, 2817.3 , 2825.79, 2835.67, 2848.94,
#        2862.06, 2873.64, 2886.52, 2903.18, 2923.29, 2939.68, 2954.94,
#        2968.62, 2979.93, 2982.26, 2983.44, 2981.64, 2986.96, 2992.26,
#        2994.11, 2992.63, 2989.66, 2984.51, 2978.32, 2971.27, 2962.5 ,
#        2955.66, 2943.29, 2929.7 , 2915.91, 2897.25, 2875.26, 2861.43,
#        2849.95, 2883.83, 2879.05, 2875.18, 2869.34, 2862.39, 2857.06,
#        2848.71, 2845.69, 2842.06, 2837.15, 2835.3 , 2832.32, 2827.56,
#        2828.99, 2831.45, 2830.56, 2828.82, 2821.72, 2808.49, 2794.98,
#        2787.79, 2783.28, 2777.83, 2772.85, 2775.95, 2770.22, 2754.77,
#        2743.85, 2731.92, 2717.59, 2714.23, 2720.63, 2722.65, 2726.09,
#        2730.46, 2733.61, 2733.72, 2732.2 , 2732.45, 2731.08, 2728.7 ,
#        2723.82, 2721.5 , 2720.8 , 2722.21, 2721.88, 2717.42, 2715.53,
#        2718.87, 2723.55, 2735.94, 2747.57, 2759.75, 2773.39, 2777.4 ,
#        2789.66, 2799.29, 2800.59, 2780.1 , 2766.28, 2754.11, 2743.13,
#        2735.98, 2726.47, 2712.25, 2698.91, 2687.08, 2672.95, 2661.45,
#        2653.56, 2647.71, 2645.49, 2643.63, 2642.83, 2641.97, 2645.14,
#        2647.52, 2647.78, 2647.92, 2650.14, 2654.24, 2662.96, 2671.81,
#        2681.91, 2697.08, 2707.39, 2717.51, 2722.86, 2724.34, 2714.78,
#        2717.08, 2722.48, 2724.14, 2723.72, 2721.23, 2717.35, 2716.4 ,
#        2721.46, 2727.29, 2724.41, 2720.42, 2719.42], dtype=np.float32)])
# arr2gt[0].shape
# arr1pred_aligned[:201].shape
# plt.plot(arr2gt[0], arr1pred_aligned[:201])

# Stage 4 -- need to use load_pred and load_dep_pred for full effect.



# Continuation of non-3DMPP stuff

frameIdx = 99
peopleBboxes = [{"bbox": x} for x in detectorOut[99][0]] # class 0 is people
pose2dOut, _ = inference_top_down_pose_model(poseModel, frames[frameIdx][:, :, ::-1], person_results=peopleBboxes, bbox_thr=0.8, format="xyxy") #keypoint locations in org [720, 1280] image coords
vis = vis_pose_result(poseModel, frames[frameIdx], pose2dOut, kpt_score_thr=0.0, radius=4, thickness=2)
plt.figure(figsize = (10,10))
plt.imshow(vis)

# testKeypoints2D = np.array([[[-0.4819], [-0.5206], [-0.5085], [-0.5375], [-0.4432], [-0.3901], [-0.3732], [-0.5013], [-0.4795], [-0.4432], [-0.4505], [-0.4408], [-0.3635], [-0.3635], [-0.5206], [-0.4916], [-0.4046]],
#  [[-0.2152], [-0.2055], [-0.0242], [ 0.1063], [-0.2248], [-0.0508], [ 0.1159], [-0.3118], [-0.4158], [-0.4278], [-0.4689], [-0.3916], [-0.3432], [-0.3771], [-0.3988], [-0.3312], [-0.3698]]]).squeeze().transpose(1, 0)
# plot2dPose(testKeypoints2D)

person1 = pose2dOut[0]
bbox = person1["bbox"]
keypointsB36M = keypointsCOCOToB36M(person1["keypoints"][:, :2])
bboxWidth = bbox[2] - bbox[0]
bboxHeight = bbox[3] - bbox[1]
keypointsNormalized = keypointsB36M[:, :2] - np.array([bbox[0], bbox[1]])[np.newaxis, :]
keypointsNormalized = keypointsNormalized / np.array([bboxWidth, bboxWidth])[np.newaxis, :]
keypointsNormalized = (keypointsNormalized - 0.5)  * 2
keypointsNormalized = keypointsNormalized * 0.23 - np.array([0.5, 0.3])[np.newaxis, :]

testKeypoints2D.shape
keypointsNormalized.shape
plot2dPose(keypointsNormalized)

#------------------------------ DynaBOA pose model ------------------------------

import os
os.chdir("/home/cameron/DynaBOA")
# from base_adaptor import BaseAdaptor
import dynaboa_internet
import utils.kp_utils

options = dynaboa_internet.parser.parse_args("""--expdir exps --expname 3dpw --dataset 3dpw --motionloss_weight 0.8 --retrieval 1 --dynamic_boa 1 --optim_steps 7 --cos_sim_threshold 3.1e-4 --shape_prior_weight 2e-4 --pose_prior_weight 1e-4 --save_res 1""".split(" "))
myDataloader = None
def set_dataloader_override(self):
	self.dataloader = myDataloader
def save_results_override(*args, **kwargs):
	print(args, kwargs)
# dynaboa_internet.BaseAdaptor.set_dataloader = set_dataloader_override
dynaboa_internet.BaseAdaptor.save_results = save_results_override
adaptor = dynaboa_internet.Adaptor(options)
adaptor.excute()
adaptor.dataloader
dataloader_output = next(enumerate(adaptor.dataloader))[1]
dataloader_output.keys()

plt.imshow((dataloader_output["image"].squeeze().numpy().transpose(1, 2, 0) + 2.2) / 2)
dataloader_output["bbox"]

dataloader_output["image"].shape
h36m = utils.kp_utils.convert_kps(dataloader_output["smpl_j2d"], "spin", "h36m")
h36m
plot2dPose(h36m[0, :, :2])

# abandoned for now, too many steps to setup / messy code

# ------------------------------ Dual Networks ------------------------------
in1 = np.array([[726.9307 , 535.13824],
	   [627.66113, 462.05984],
	   [754.58734, 555.49805],
	   [726.3472 , 534.7087 ],
	   [805.20355, 592.7597 ],
	   [899.5302 , 662.1993 ],
	   [840.3574 , 618.6386 ],
	   [670.67426, 493.72437],
	   [774.1009 , 569.86316],
	   [809.55615, 595.9638 ],
	   [757.77094, 557.8417 ],
	   [702.81305, 517.3837 ],
	   [848.68115, 624.7663 ],
	   [787.081  , 579.4185 ],
	   [650.6255 , 478.96524],
	   [698.8398 , 514.4588 ],
	   [636.80493, 468.79114]])
in2 = np.array([[752.07025, 583.1933 ],
	   [712.2081 , 571.10986],
	   [689.0743 , 706.80096],
	   [676.8739 , 787.62683],
	   [793.1561 , 591.71313],
	   [787.1575 , 724.9936 ],
	   [808.67975, 794.7828 ],
	   [768.34875, 510.54123],
	   [766.2095 , 431.42923],
	   [758.2116 , 406.92233],
	   [754.9545 , 344.8456 ],
	   [812.3987 , 452.24612],
	   [807.58716, 538.6665 ],
	   [750.24664, 489.86526],
	   [715.0468 , 441.03467],
	   [681.8384 , 518.0152 ],
	   [683.50336, 468.39264]])
plt.scatter(in1[:, 0], in1[:, 1])
plot2dPose(in2)

!ls
os.chdir("/home/cameron/3D-Multi-Person-Pose/mupots/est_p2ds")
estP2d1 = np.load("00_00.pkl", allow_pickle=True)
# estP2d2 = np.load("00_01.pkl", allow_pickle=True)

plot2dPose(estP2d1[0][80])
# plot2dPose(estP2d2[0][80])


estP2d1[1][0]
estP2d[1][0][0]
plt.hist(estP2d[1].ravel())
plt.hist(estP2d[1].ravel())

os.chdir("/home/cameron/3D-Multi-Person-Pose/mupots/est_p2ds")
estP2d1 = np.load("00_01.pkl", allow_pickle=True)
plot2dPose(estP2d1[0][0])
os.chdir("/home/cameron/3DMPP_Modified/mupots/est_p2ds")
estP2d1 = np.load("0_Custom_00_00.pkl", allow_pickle=True)
plot2dPose(estP2d1[0][0])


#------------------------------ STMO POSE LIFTER ------------------------------

from model.stmo import Model as STMOModel
from common.opt import opts as STMOOptions
#
def loadPoseLifter():
	os.chdir("/home/cameron/p-stmo")
	optionsParser = STMOOptions()
	optionsParser.init()
	optionsStart = optionsParser.parser.parse_args(["-f", "243", "--reload", "1", "-tds", "2", "--layers", "4", "--previous_dir", "checkpoint/PSTMO_no_refine_11_4288_h36m_cpn.pth"])
	optionsParser.init = lambda: None
	optionsParser.parser.parse_args = lambda: optionsStart
	opts = optionsParser.parse()

	stmoModel = torch.nn.DataParallel(STMOModel(opts))
	pre_dict = torch.load(opts.previous_dir)
	stmoModel.load_state_dict(pre_dict)
	stmoModel.eval() #input is [B, coordDim(2), frames(243), keypointId(17), 1]
	# normalize keypoints with X / w * 2 - [1, h / w]
	return stmoModel

# testInput = testKeypoints2D.transpose(1, 0)[np.newaxis, :, np.newaxis, :, np.newaxis]
testInput = keypointsNormalized.transpose(1, 0)[np.newaxis, :, np.newaxis, :, np.newaxis]
testInput.shape
testInput = testInput.repeat(243, 2)
testInput.shape
testInput = torch.tensor(testInput).type(torch.float32)

stmoModel = loadPoseLifter()
pose3DOut = stmoModel(testInput)
keypoints3D = pose3DOut[0].squeeze().cpu().detach().numpy().transpose(1, 0)
adjKeypoints3D = keypoints3D[:, (0, 2, 1)]
adjKeypoints3D[:, 2] = -adjKeypoints3D[:, 2]
# always leaning forward, think it has something to do with the camera transform being different. firstly, uncropped video has diff camera transform than cropped b36m image. Secondly, cropped bounding of test data will have weird camera transforms, way diff from test data. Soln is 1. for each boundign box, to somehow hallucinate it as if it was taken with the h36m camera. Or 2. models trained on wide variety of camera params. Prob h36m -> 3d pose keypoints -> augmented with many different camera transforms (at least capturing what a phone camera is like) is way to go.
plot3dPose(adjKeypoints3D)

# GROUND TRUTH DATA not skewed like results were getting
gt2DData = dict(np.load("/home/cameron/p-stmo/dataset/data_2d_h36m_gt.npz", allow_pickle=True))
gt3DData = dict(np.load("/home/cameron/p-stmo/dataset/data_3d_h36m.npz", allow_pickle=True))
gt3DData["positions_3d"][0]
gt3DData["positions_3d"].item().keys()
gt3DData["positions_3d"].item()["S1"].keys()
jointKeys = list(set(range(32)) - {4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31})
walking3DGt = gt3DData["positions_3d"].item()["S1"]["Walking 1"][:, jointKeys, :]
walking3DGt.shape
[plot3dPose(walking3DGt[i*100]) for i in range(10)]


# look at poseAnalyzer1 for how to do temporal padding to 243 frames

#------------------------------ UTILITY ------------------------------

def traverseHierarchy(h, startKey, callback):
	parentId = startKey
	if type(h) is dict:
		children = h.keys()
		for childKey in children:
			callback(parentId, childKey)
			traverseHierarchy(h[childKey], childKey, callback)
	else:
		childKey = h
		callback(parentId, childKey)
def plot3dPose(keypoints3d):
	# in keypoints, Y(second dim) is up, and Y moves down.

	# keypoint hierarchy starting at root
	hierarchy = {1: {2: 3}, 4: {5: 6}, 7: {8: {9: 10, 11: {12: 13}, 14: {15: 16}}}}
	# calls callback(id1, id2) for every bone between joints
	xs = []
	ys = []
	zs = []
	colors = []
	def addToPlot(parentId, childId):
		p1 = keypoints3d[parentId]
		p2 = keypoints3d[childId]
		xs.extend([p1[0], p2[0]])
		ys.extend([p1[2], p2[2]]) #swap y and z, make down->up up->down
		zs.extend([-p1[1], -p2[1]])
		boneId = str(parentId) + "-" + str(childId)
		colors.extend([boneId, boneId])

	traverseHierarchy(hierarchy, 0, addToPlot)

	fig = px.line_3d(x=xs, y=ys, z=zs, color=colors)
	fig.update_layout(scene_aspectmode='data')
	fig.show()

def plot2dPose(keypoints2d, overlay=False):
	hierarchy = {1: {2: 3}, 4: {5: 6}, 7: {8: {9: 10, 11: {12: 13}, 14: {15: 16}}}}
	# calls callback(id1, id2) for every bone between joints
	def traverseHierarchy(h, startKey, callback):
		parentId = startKey
		if type(h) is dict:
			children = h.keys()
			for childKey in children:
				callback(parentId, childKey)
				traverseHierarchy(h[childKey], childKey, callback)
		else:
			childKey = h
			callback(parentId, childKey)

	def addToPlot(parentId, childId):
		p1 = keypoints2d[parentId]
		p2 = keypoints2d[childId]
		plt.plot([p1[0], p2[0]], [-p1[1], -p2[1]])

	if not overlay:
		plt.figure(figsize = (3,7))
	traverseHierarchy(hierarchy, 0, addToPlot)

# # CONFIG PIPELINE AND INFERENCE
# import mmcv
# from mmdet.datasets.pipelines import Compose
# from mmdet.datasets import replace_ImageToTensor
# config = mmcv.Config.fromfile(modelConfig)
# config.data.test.pipeline[0].type = 'LoadImageFromWebcam'
# config.data.test.pipeline = replace_ImageToTensor(config.data.test.pipeline)
# test_pipeline = Compose(config.data.test.pipeline)
# pipelineInput = dict(img=frames[0, :, :, ::-1])
# pipelineOutput = test_pipeline(pipelineInput)
# # plt.imshow(pipelineOutput["img"][0].data.numpy().transpose(1, 2, 0).astype(np.uint8))
# img = pipelineOutput["img"][0].data[None, :, :, :].to(device)
# img_metas = pipelineOutput["img_metas"][0].data
# # things like ori_shape and whatnot in img_metas
# with torch.no_grad():
#     out = detectorModule.forward_test([img], [[img_metas]],rescale=True)
# resVis = model.show_result(frames[0, :, :, ::1], out[0], score_thr=0.8, bbox_color=None, text_color=(200, 200, 200), mask_color=None)
# plt.imshow(resVis)
#
# img_metas

# REFERENCE INFERENCE DETECTOR
# outRef = mmdet.apis.inference_detector(detectorModule, frames[0, :, :, ::-1])
# resVis = detectorModule.show_result(frames[0, :, :, ::1], outRef, score_thr=0.8, bbox_color=None, text_color=(200, 200, 200), mask_color=None)
# plt.imshow(resVis)

# # REFERENCE INFERENCE POSE
# from mmpose.apis import process_mmdet_results
# outRef[0]
# detectorOut[0][0]
# keypoints, _ = inference_top_down_pose_model(poseModel, frames[0][:, :, ::-1], person_results=process_mmdet_results(outRef, 1), bbox_thr=0.5, format="xyxy")
# peopleBboxes[:5]
# process_mmdet_results(outRef, 1)[:5]
# vis = vis_pose_result(poseModel, processedFrames[0].numpy().transpose(1, 2, 0)[:, :, ::-1], keypoints, kpt_score_thr=0.0, radius=4, thickness=2)
# plt.figure(figsize = (10,10))
# plt.imshow(vis)
#
