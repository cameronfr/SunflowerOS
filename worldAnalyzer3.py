# Yolox-l + vitpose + 3Dcrowdnet

import torch
import torchvision
import numpy as np
import sys
import mmdet.apis
import os
import av
from PIL import Image

import plotly.express as px
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams["axes.grid"] = False
# %config InlineBackend.figure_format='svg'
%config InlineBackend.figure_format = 'retina'
plt.rcParams['figure.figsize'] = [12.0, 8.0]

device = torch.cuda.current_device()

# ------------------------------ HELPERS------------------------------

class DictToObject(object):
	def __init__(self, d):
		for a, b in d.items():
			if isinstance(b, (list, tuple)):
			   setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
			else:
			   setattr(self, a, obj(b) if isinstance(b, dict) else b)

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

# Plots take in B36m format
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

# ------------------------------ YOLOX DETECTOR ------------------------------

# in the mmdet Repo
def loadDetector():
	os.chdir("/home/cameron/mmdetection/")
	modelConfig = "configs/yolox/yolox_x_8x8_300e_coco.py"
	modelCheckpoint = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth"
	model = mmdet.apis.init_detector(modelConfig, modelCheckpoint, device) # this is a torch module.
	sum([x.numel() for x in model.parameters()])
	return model

detectorModule = loadDetector()
def detector2D(frames):
	detectorOut = []
	for frame in frames:
		detectorOut.append(mmdet.apis.inference_detector(detectorModule, frame[:, :, ::-1]))
	return detectorOut

# ------------------------------ ViTPose ------------------------------
# uses frames and detectorOut from previous step

sys.path.insert(0, "/home/cameron/vitpose/") # load the modified mmpose from the vitpose repo. Modifiy mmpose/mmpose/__init__.py so that "1.5.0" max is replaced w/ current install
import mmpose.apis
# from mmpose.apis import inference_top_down_pose_model, vis_pose_result, get_track_id # its messy / not good structure, resist urge to clean up
import copy
import tqdm

def loadPose2DModel():
	os.chdir("/home/cameron/")
	modelConfig = "vitpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py"
	modelCheckpoint = "vitpose-l-multi-coco.pth" #download from onedrive
	model = mmpose.apis.init_pose_model(modelConfig, modelCheckpoint, device) # this is a torch module.
	sum([x.numel() for x in model.parameters()])
	return model
poseModel = loadPose2DModel()

def pose2D(frame, detections2D, lastPose2dOutWithTrackId, nextTrackId, score_thr=0.5):
	people = detections2D[0]
	confidences = people[:, 4]
	peopleBboxes = [{"bbox": x} for x in people[confidences > score_thr]]
	pose2dOut, _ = mmpose.apis.inference_top_down_pose_model(poseModel, frames[frameIdx][:, :, ::-1], person_results=peopleBboxes, bbox_thr=0.0, format="xyxy")

	# pose2dOut is list of {"keypoints" ..., "bbox": ...}, and pose2dOutWithTrackId adds a "track id" to each dict.
	pose2dOutWithTrackId, nextTrackId = mmpose.apis.get_track_id(pose2dOut, copy.deepcopy(lastPose2dOutWithTrackId), nextTrackId, use_oks=True, tracking_thr=0.3) # code deletes objs in last pose...
	return (pose2dOutWithTrackId, nextTrackId)

# posesByTrackId = {}
# for frameIdx, poses in enumerate(posesByFrame):
# 	for pose in poses:
# 		track_id = pose["track_id"]
# 		if track_id not in posesByTrackId:
# 			posesByTrackId[track_id] = []
# 		else:
# 			if posesByTrackId[track_id][-1]["frameIdx"] != frameIdx - 1:
# 				raise "track is non contiguous, not expected"
# 		pose["frameIdx"] = frameIdx
# 		posesByTrackId[track_id].append(pose)

# ------------------------------ CrowdPose3D ------------------------------

os.chdir("/home/cameron/Crowdnet3D/demo")
import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import colorsys
import json
import random
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import process_bbox, generate_patch_image, get_bbox
from utils.transforms import pixel2cam, cam2pixel, transform_joint_to_other_db
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"
from utils.vis import vis_mesh, save_obj, render_mesh, vis_coco_skeleton
sys.path.insert(0, cfg.smpl_path)
from utils.smpl import SMPL

def add_pelvis(joint_coord, joints_name):
	lhip_idx = joints_name.index('L_Hip')
	rhip_idx = joints_name.index('R_Hip')
	pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
	pelvis[2] = joint_coord[lhip_idx, 2] * joint_coord[rhip_idx, 2]  # confidence for openpose
	pelvis = pelvis.reshape(1, 3)

	joint_coord = np.concatenate((joint_coord, pelvis))

	return joint_coord
def add_neck(joint_coord, joints_name):
	lshoulder_idx = joints_name.index('L_Shoulder')
	rshoulder_idx = joints_name.index('R_Shoulder')
	neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
	neck[2] = joint_coord[lshoulder_idx, 2] * joint_coord[rshoulder_idx, 2]
	neck = neck.reshape(1,3)

	joint_coord = np.concatenate((joint_coord, neck))

	return joint_coord

# cfg.set_args(args.gpu_ids, is_test=True)
cfg.set_args("0", is_test=True)
cfg.render = True
cudnn.benchmark = True

# SMPL joint set
joint_num = 30  # original: 24. manually add nose, L/R eye, L/R ear, head top
joints_name = (
'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe',
'Neck', 'L_Thorax', 'R_Thorax',
'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose', 'L_Eye',
'R_Eye', 'L_Ear', 'R_Ear', 'Head_top')
flip_pairs = (
(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28))
skeleton = (
(0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17), (17, 19),
(19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 24), (24, 15), (24, 25), (24, 26),
(25, 27), (26, 28), (24, 29))

# SMPl mesh
vertex_num = 6890
smpl = SMPL()
face = smpl.face

# other joint set
coco_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')
coco_skeleton = (
(1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 6),
(11, 17), (12,17), (17,18))

vis_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Thorax', 'Pelvis')
vis_skeleton = ((0, 1), (0, 2), (2, 4), (1, 3), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 17), (6, 17), (11, 18), (12, 18), (17, 18), (17, 0), (6, 8), (8, 10),)

# snapshot load
model_path = "demo_checkpoint.pth.tar"
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model(vertex_num, joint_num, 'test')

model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

transform = transforms.ToTensor()

# for frameIdx, frame in tqdm.tqdm(enumerate(frames[:])):
def multiPose3D(frame, poses2D):
	# img_name = f"""frame{frameIdx}.jpg"""
	# img_path = img_name
	# coco_joint_list = [pose["keypoints"].tolist() for pose in posesByFrame[frameIdx]]
	coco_joint_list = [pose["keypoints"].tolist() for pose in poses2D]
	image = frame
	original_img = image[:, :, ::-1]
	input = original_img.copy()
	input2 = original_img.copy()
	original_img_height, original_img_width = original_img.shape[:2]

	drawn_joints = []
	c = coco_joint_list
	# manually assign the order of output meshes
	# coco_joint_list = [c[2], c[0], c[1], c[4], c[3]]

	allOuts = []
	for idx in range(len(coco_joint_list)):
		""" 2D pose input setting & hard-coding for filtering """
		pose_thr = 0.1
		coco_joint_img = np.asarray(coco_joint_list[idx])[:, :3]
		coco_joint_img = add_pelvis(coco_joint_img, coco_joints_name)
		coco_joint_img = add_neck(coco_joint_img, coco_joints_name)
		coco_joint_valid = (coco_joint_img[:, 2].copy().reshape(-1, 1) > pose_thr).astype(np.float32)

		# filter inaccurate inputs
		det_score = sum(coco_joint_img[:, 2])
		if det_score < 1.0:
			print("Filtered out by sum of joint probs")
			continue
		if len(coco_joint_img[:, 2:].nonzero()[0]) < 1:
			print("Filtered out by joints being zero")
			continue
		# filter the same targets
		tmp_joint_img = coco_joint_img.copy()
		continue_check = False
		for ddx in range(len(drawn_joints)):
			drawn_joint_img = drawn_joints[ddx]
			drawn_joint_val = (drawn_joint_img[:, 2].copy().reshape(-1, 1) > pose_thr).astype(np.float32)
			diff = np.abs(tmp_joint_img[:, :2] - drawn_joint_img[:, :2]) * coco_joint_valid * drawn_joint_val
			diff = diff[diff != 0]
			if diff.size == 0:
				continue_check = True
			elif diff.mean() < 20:
				continue_check = True
		if continue_check:
			print("(disabled) Filtered out by drawn_joints check")
			# continue
		drawn_joints.append(tmp_joint_img)

		""" Prepare model input """
		# prepare bbox
		bbox = get_bbox(coco_joint_img, coco_joint_valid[:, 0]) # xmin, ymin, width, height
		bbox = process_bbox(bbox, original_img_width, original_img_height)
		if bbox is None:
			continue
		img, img2bb_trans, bb2img_trans = generate_patch_image(input2[:,:,::-1], bbox, 1.0, 0.0, False, cfg.input_img_shape)
		img = transform(img.astype(np.float32))/255
		img = img.cuda()[None,:,:,:]

		coco_joint_img_xy1 = np.concatenate((coco_joint_img[:, :2], np.ones_like(coco_joint_img[:, :1])), 1)
		coco_joint_img[:, :2] = np.dot(img2bb_trans, coco_joint_img_xy1.transpose(1, 0)).transpose(1, 0)
		coco_joint_img[:, 0] = coco_joint_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
		coco_joint_img[:, 1] = coco_joint_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

		coco_joint_img = transform_joint_to_other_db(coco_joint_img, coco_joints_name, joints_name)
		coco_joint_valid = transform_joint_to_other_db(coco_joint_valid, coco_joints_name, joints_name)
		coco_joint_valid[coco_joint_img[:, 2] <= pose_thr] = 0

		# check truncation
		coco_joint_trunc = coco_joint_valid * ((coco_joint_img[:, 0] >= 0) * (coco_joint_img[:, 0] < cfg.output_hm_shape[2]) * (coco_joint_img[:, 1] >= 0) * (coco_joint_img[:, 1] < cfg.output_hm_shape[1])).reshape(
			-1, 1).astype(np.float32)
		coco_joint_img, coco_joint_trunc, bbox = torch.from_numpy(coco_joint_img).cuda()[None, :, :], torch.from_numpy(coco_joint_trunc).cuda()[None, :, :], torch.from_numpy(bbox).cuda()[None, :]

		""" Model forward """
		inputs = {'img': img, 'joints': coco_joint_img, 'joints_mask': coco_joint_trunc}
		targets = {}
		meta_info = {'bbox': bbox}
		with torch.no_grad():
			out = model(inputs, targets, meta_info, 'test')
		allOuts.append(out)

		# draw output mesh
		mesh_cam_render = out['mesh_cam_render'][0].cpu().numpy()
		bbox = out['bbox'][0].cpu().numpy()
		princpt = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
		# original_img = vis_bbox(original_img, bbox, alpha=1)  # for debug

		# generate random color
		color = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)
		original_img = render_mesh(original_img, mesh_cam_render, face, {'focal': cfg.focal, 'princpt': princpt}, color=color)
		# plt.imshow(original_img[:, :, ::-1])
		# plt.show()

		# Save output mesh
		# output_dir = 'output_custom'
		# file_name = f'{output_dir}/{img_path.split("/")[-1][:-4]}_{idx}.jpg'
	# print("file name: ", file_name)
		# save_obj(mesh_cam_render, face, file_name=f'{output_dir}/{img_path.split("/")[-1][:-4]}_{idx}.obj')
	# cv2.imwrite(file_name, original_img)
	return (allOuts, original_img[:, :, ::-1])

		# Draw input 2d pose
		# tmp_joint_img[-1], tmp_joint_img[-2] = tmp_joint_img[-2].copy(), tmp_joint_img[-1].copy()
		# input = vis_coco_skeleton(input, tmp_joint_img.T, vis_skeleton)
		# cv2.imwrite(file_name[:-4] + '_2dpose.jpg', input)


# ------------------------------ Load Video ------------------------------


# videoRaw = av.open("/home/cameron/shibuyaTrim1.mp4")
videoRaw = av.open("/home/cameron/Crowdnet3D/squidGame.mp4")
# videoRaw.seek(3*10**6)
videoStream = videoRaw.streams[0]
videoFrames = videoRaw.decode(videoRaw.streams[0])
videoArray = np.empty((videoStream.frames, videoStream.height, videoStream.width, 3), dtype=np.uint8) #585MB for 10 seconds of video
for idx, frame in enumerate(videoRaw.decode(videoRaw.streams[0])):
	videoArray[idx] = np.array(frame.to_image())
actualFramecount = int(videoStream.base_rate * videoStream.duration * videoStream.time_base)
videoArray = videoArray[:actualFramecount]

# frames = np.array(Image.open("/home/cameron/Crowdnet3D/demo/input/images/100023.jpg"))[np.newaxis, ...]
frames = videoArray[:]#[0:100]

# ------------------------------ CONNECT TO RENDERER ------------------------------

import zmq
context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.bind('tcp://*:5558')
# img = cv2.flip(frames[0], 0).ravel()
# message = np.insert(img, 0, 0)
# socket.send(message.tobytes())
#
def sendCommand(id, data):
	message = np.array([id], dtype=np.uint8).tobytes() + data
	socket.send(message)


# Coordinate space: y (2nd coord) is up, z away from camera. T-pose facing towards camera.
jointSpecCrowdnet = {"names": None, "skeleton": None, "tpose": None}
jointSpecCrowdnet["names"] = ['Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'Head_Top']
jointSpecCrowdnet["skeleton"] = [(0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17), (17, 19), (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 24), (24, 15), (24, 25), (24, 26), (25, 27), (26, 28), (24, 29)]

jointSpecUnity = {"names": None, "skeleton": None, "tpose": None, "unityNames": None}
jointSpecUnity["names"] = ['Pelvis',  'L_Hip', 'L_Knee', 'L_Ankle', 'R_Hip', 'R_Knee', 'R_Ankle', 'Spine', 'Chest', 'Neck', 'Head', 'Left_Eye', 'Right_Eye', 'Left_Shoulder', 'Left_Elbow', 'Left_Wrist', 'Right_Shoulder', 'Right_Elbow', 'Right_Wrist']
jointSpecUnity["unityNames"] = ["Hips", "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "RightUpperLeg", "RightLowerLeg", "RightFoot", "Spine", "Chest", "Neck", "Head", "LeftEye", "RightEye", "LeftUpperArm", "LeftLowerArm", "LeftHand", "RightUpperArm", "RightLowerArm", "RightHand"] # HumanBodyBones names
jointSpecUnity["skeleton"] = [(0, 1), (1,2), (2,3), (0,4), (4,5), (5,6), (0, 7), (7,8), (8,9), (9,10), (10, 11), (10, 12), (8, 13), (13,14), (14,15), (8,16), (16,17), (17,18)]
jointSpecUnity["tpose"] = [[1, 0, 0], [0, -1, 0], [0, -1, 0], [-1, 0, 0], [0, -1, 0], [0, -1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0], [-1, 1, 0], [-1, 0, 0], [-1, 0, 0]]
for idx, pair in enumerate(jointSpecUnity["skeleton"]):
	print(jointSpecUnity["unityNames"][pair[0]], jointSpecUnity["unityNames"][pair[1]])
	print(jointSpecUnity["tpose"][idx])

# Transform from x, y-down, z-forwards to unity (x, y-up, z-forwards). Expects last dim to be size-3
def coordsToUnity(arr):
	assert arr.shape[-1] == 3
	newArr = np.zeros(arr.shape)
	newArr[..., 0] = arr[..., 0]
	newArr[..., 1] = -arr[..., 1]
	newArr[..., 2] = arr[..., 2]
	return newArr

# ------------------------------ RUN PROCESS ------------------------------

lastPose2dOutWithTrackId = []
nextTrackId = 0 # Same track ID in a detection <-> same person. nextTrackId increases each time new person detected.
# for frameIdx in range(len(frames)):
for frameIdx in range(60,61, 10):
	# frameIdx = 61
	frame = frames[frameIdx]
	sendCommand(0, cv2.flip(frame, 0).ravel().tobytes())

	detection2D = detector2D([frame])[0]
	targetVisClass = 0
	# mmdet.apis.show_result_pyplot(detectorModule, frame[:, :, ::-1], detection2D, score_thr=0.5)
	vis = detectorModule.show_result(frame, detection2D, score_thr=0.4)
	plt.figure(figsize = (15,15))
	plt.imshow(vis)

	# len(list(filter(lambda x: x[4] > 0.4, detection2D[0].tolist())))
	pose2dOutWithTrackId, nextTrackId = pose2D(frame, detection2D, lastPose2dOutWithTrackId, nextTrackId, score_thr=0.4)
	lastPose2dOutWithTrackId = pose2dOutWithTrackId

	vis = mmpose.apis.vis_pose_tracking_result(poseModel, frame, pose2dOutWithTrackId, kpt_score_thr=0.3, radius=4, thickness=2)
	plt.figure(figsize = (15,15))
	plt.imshow(vis)

	multiPose3DOuts, vis = multiPose3D(frame, pose2dOutWithTrackId)
	for i in range(len(pose2dOutWithTrackId)):
		multiPose3DOuts[i]["track_id"] = pose2dOutWithTrackId[i]["track_id"]
	plt.figure(figsize=(15, 15))
	plt.imshow(vis)

	allJointsList = []
	for multiPose3DOut in multiPose3DOuts[:]:
		bbox = multiPose3DOut['bbox'][0].cpu().numpy()
		princpt = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
		# Coords for this person are rendered using. Where c_x_i and c_y_i change per person
		# U_1 = (f_x_1 * x + c_x_i * z) / z
		# V_1 = (f_y_1 * y + c_y_i * z) / z
		# Want to adjust coords so that we get the same render result with fixed camera with (640, 360) optical center. Assuming this is author's intended space (i.e. not changing z), are there are many possible inverse projections if alter z?
		# U_2 = (f_x * x' + (img_width/2) * z') / z'
		# V_2 = (f_y * y' + (img_height/2) * z') / z'
		# Then, to get same U_1 and U_2 when c_x and c_y are 0,
		# (f_x_1 * x + c_x_i * z) / z = (f_x * x' + img_height/2 * z') / z' => f_x_1 * x + c_x_i * z = f_x * x' + (img_width/2) * z => x' = (f_x_1 * x + c_x_i * z - z*(img_width/2)) / f_x => x' = (f_x *x + z * (c_x_i - img_width/2)) / f_x = x + (z * (c_x_i - img_width/2)) / f_x
		# And similarly y' = y + (z * (c_y_i - img_width/2)) / f_y
		# z' = z, f_x = f_x_1, f_y = f_y_1
		# So now, if multiply focal length 5000 camera matrix, will get correct projection to ([-640, 640], [-360, 360])
		# Next, need to adjust coordinates so that they project in centered way
		cameraIntrinsics = np.array([ [cfg.focal[0], 0, 0], [0, cfg.focal[1], 0], [0,0,1]])
		jointsWorld = multiPose3DOut["joint_cam"].squeeze().cpu().numpy().copy()
		jointsWorld[:, 0] = jointsWorld[:, 0] + (((princpt[0] - frame.shape[1]/2) * jointsWorld[:, 2])/cfg.focal[0])
		jointsWorld[:, 1] = jointsWorld[:, 1] + (((princpt[1] - frame.shape[0]/2) * jointsWorld[:, 2])/cfg.focal[1])
		pos = cameraIntrinsics @ jointsWorld.T
		projectedJointCam = pos.T / pos.T[:, 2, np.newaxis]
		allJointsList.append(jointsWorld)

		posTargetUnity = coordsToUnity(jointsWorld[0])
		trackId = multiPose3DOut["track_id"]
		sendCommand(1, np.array([trackId], dtype=np.int32).tobytes()) # spawn a character
		sendCommand(2, np.array([trackId], dtype=np.int32).tobytes() + posTargetUnity.astype(np.float32).tobytes())
		# plt.imshow(frame)
		# plt.scatter(princpt[0], princpt[1])
		# plt.scatter(*(projectedJointCam[:, :2] + np.array([640, 360])[np.newaxis, :]).transpose(1, 0))

	# In Unity: set image to fill back of camera (with correct aspect ratio, with height filling full height of screen). Then set vertical FOV to this calc fov.
	cameraVerticalFovDeg = 2 * np.arctan((0.5*frame.shape[0]) / cfg.focal[1]) * (180/np.pi)

	#Format: dim 1 is up-down and is reversed (bigger is lower).
	# allJoints = np.stack(allJointsList).reshape(-1, 3)
	# plt.scatter(*allJoints[:, :2].transpose(1, 0))
	# fig = px.scatter_3d(x=allJoints[:, 0], y=allJoints[:, 2], z=-allJoints[:, 1])
	# fig.update_traces(marker={'size': 1})
	# fig.update_layout(scene_aspectmode='data')
	# fig.show()
