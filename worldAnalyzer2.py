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

frames = videoArray[0:100]

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

from mmpose.apis import inference_top_down_pose_model, vis_pose_result # its messy / not good structure, resist urge to clean up

poseModel = loadPose2D()
frameIdx = 99
peopleBboxes = [{"bbox": x} for x in detectorOut[99][0]] # class 0 is people
pose2dOut, _ = inference_top_down_pose_model(poseModel, frames[frameIdx][:, :, ::-1], person_results=peopleBboxes, bbox_thr=0.5, format="xyxy") #keypoint locations in org [720, 1280] image coords
vis = vis_pose_result(poseModel, frames[frameIdx], pose2dOut, kpt_score_thr=0.0, radius=4, thickness=2)
plt.figure(figsize = (10,10))
plt.imshow(vis)

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
		ys.extend([p1[1], p2[1]])
		zs.extend([p1[2], p2[2]])
		boneId = str(parentId) + "-" + str(childId)
		colors.extend([boneId, boneId])

	traverseHierarchy(hierarchy, 0, addToPlot)

	fig = px.line_3d(x=xs, y=ys, z=zs, color=colors)
	fig.update_layout(scene_aspectmode='data')
	fig.show()

def plot2dPose(keypoints2d):
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
