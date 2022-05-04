import matplotlib.pyplot as plt
plt.style.use("ggplot")
%config InlineBackend.figure_format='svg'

# https://mmpose.readthedocs.io/en/latest/api.html#mmpose.apis.inference_top_down_pose_model
# https://vinnik-dmitry07.medium.com/a-python-unity-interface-with-zeromq-12720d6b7288

# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
from argparse import ArgumentParser

import cv2
import mmcv
import numpy as np

from mmpose.apis import (extract_pose_sequence, get_track_id,
						 inference_pose_lifter_model,
						 inference_top_down_pose_model, init_pose_model,
						 process_mmdet_results, vis_3d_pose_result)
from mmpose.core import Smoother

try:
	from mmdet.apis import inference_detector, init_detector

	has_mmdet = True
except (ImportError, ModuleNotFoundError):
	has_mmdet = False


def convert_keypoint_definition(keypoints, pose_det_dataset,
								pose_lift_dataset):
	"""Convert pose det dataset keypoints definition to pose lifter dataset
	keypoints definition.

	Args:
		keypoints (ndarray[K, 2 or 3]): 2D keypoints to be transformed.
		pose_det_dataset, (str): Name of the dataset for 2D pose detector.
		pose_lift_dataset (str): Name of the dataset for pose lifter model.
	"""
	if pose_det_dataset == 'TopDownH36MDataset' and \
			pose_lift_dataset == 'Body3DH36MDataset':
		return keypoints
	elif pose_det_dataset == 'TopDownCocoDataset' and \
			pose_lift_dataset == 'Body3DH36MDataset':
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
	else:
		raise NotImplementedError


# def main():
	# parser = ArgumentParser()
	# parser.add_argument('det_config', help='Config file for detection')
	# parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
	# parser.add_argument(
	# 	'pose_detector_config',
	# 	type=str,
	# 	default=None,
	# 	help='Config file for the 1st stage 2D pose detector')
	# parser.add_argument(
	# 	'pose_detector_checkpoint',
	# 	type=str,
	# 	default=None,
	# 	help='Checkpoint file for the 1st stage 2D pose detector')
	# parser.add_argument(
	# 	'pose_lifter_config',
	# 	help='Config file for the 2nd stage pose lifter model')
	# parser.add_argument(
	# 	'pose_lifter_checkpoint',
	# 	help='Checkpoint file for the 2nd stage pose lifter model')
	# parser.add_argument(
	# 	'--video-path', type=str, default='', help='Video path')
	# parser.add_argument(
	# 	'--rebase-keypoint-height',
	# 	action='store_true',
	# 	help='Rebase the predicted 3D pose so its lowest keypoint has a '
	# 	'height of 0 (landing on the ground). This is useful for '
	# 	'visualization when the model do not predict the global position '
	# 	'of the 3D pose.')
	# parser.add_argument(
	# 	'--norm-pose-2d',
	# 	action='store_true',
	# 	help='Scale the bbox (along with the 2D pose) to the average bbox '
	# 	'scale of the dataset, and move the bbox (along with the 2D pose) to '
	# 	'the average bbox center of the dataset. This is useful when bbox '
	# 	'is small, especially in multi-person scenarios.')
	# parser.add_argument(
	# 	'--num-instances',
	# 	type=int,
	# 	default=-1,
	# 	help='The number of 3D poses to be visualized in every frame. If '
	# 	'less than 0, it will be set to the number of pose results in the '
	# 	'first frame.')
	# parser.add_argument(
	# 	'--show',
	# 	action='store_true',
	# 	default=False,
	# 	help='whether to show visualizations.')
	# parser.add_argument(
	# 	'--out-video-root',
	# 	type=str,
	# 	default='vis_results',
	# 	help='Root of the output video file. '
	# 	'Default not saving the visualization video.')
	# parser.add_argument(
	# 	'--device', default='cuda:0', help='Device for inference')
	# parser.add_argument(
	# 	'--det-cat-id',
	# 	type=int,
	# 	default=1,
	# 	help='Category id for bounding box detection model')
	# parser.add_argument(
	# 	'--bbox-thr',
	# 	type=float,
	# 	default=0.9,
	# 	help='Bounding box score threshold')
	# parser.add_argument('--kpt-thr', type=float, default=0.3)
	# parser.add_argument(
	# 	'--use-oks-tracking', action='store_true', help='Using OKS tracking')
	# parser.add_argument(
	# 	'--tracking-thr', type=float, default=0.3, help='Tracking threshold')
	# parser.add_argument(
	# 	'--radius',
	# 	type=int,
	# 	default=8,
	# 	help='Keypoint radius for visualization')
	# parser.add_argument(
	# 	'--thickness',
	# 	type=int,
	# 	default=2,
	# 	help='Link thickness for visualization')
	# parser.add_argument(
	# 	'--smooth',
	# 	action='store_true',
	# 	help='Apply a temporal filter to smooth the pose estimation results. '
	# 	'See also --smooth-filter-cfg.')
	# parser.add_argument(
	# 	'--smooth-filter-cfg',
	# 	type=str,
	# 	default='configs/_base_/filters/one_euro.py',
	# 	help='Config file of the filter to smooth the pose estimation '
	# 	'results. See also --smooth.')

def get3dPosesFromVideo(videoPath, numFrames=-1):
	os.chdir("/Users/cameronfranz/Desktop/avatarExperiment1/mmpose")
	assert has_mmdet, 'Please install mmdet to run the demo.'

	class DictObj:
		def __init__(self, in_dict:dict):
			assert isinstance(in_dict, dict)
			for key, val in in_dict.items():
				if isinstance(val, (list, tuple)):
				   setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
				else:
				   setattr(self, key, DictObj(val) if isinstance(val, dict) else val)

	args = DictObj({
		"det_config": "demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py",
		"det_checkpoint": "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
		"pose_detector_config": "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py",
		"pose_detector_checkpoint": "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth",
		"pose_lifter_config": "configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_243frames_fullconv_supervised_cpn_ft.py",
		# "pose_lifter_checkpoint": "https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth",
		"pose_lifter_checkpoint": "https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised-880bea25_20210527.pth",
		#extra options
		"video_path": videoPath,
		"out_video_root": "vis_results",
		"rebase_keypoint_height": True,
		"device": "CPU",
		#defaults
		"show": False,
		"det_cat_id": 1,
		"bbox_thr": 0.9,
		"use_oks_tracking": True,
		"tracking_thr": 0.3,
		"smooth": True,
		"smooth_filter_cfg": "configs/_base_/filters/one_euro.py",
		"num_instances": 1,
		"norm_pose_2d": True,
		"radius": 8,
		"thickness": 2,
	})


	# args = parser.parse_args()
	assert args.show or (args.out_video_root != '')
	assert args.det_config is not None
	assert args.det_checkpoint is not None

	video = mmcv.VideoReader(args.video_path)
	assert video.opened, f'Failed to load video file {args.video_path}'

	# First stage: 2D pose detection
	print('Stage 1: 2D pose detection.')

	person_det_model = init_detector(
		args.det_config, args.det_checkpoint, device=args.device.lower())

	pose_det_model = init_pose_model(
		args.pose_detector_config,
		args.pose_detector_checkpoint,
		device=args.device.lower())

	assert pose_det_model.cfg.model.type == 'TopDown', 'Only "TopDown"' \
		'model is supported for the 1st stage (2D pose detection)'

	pose_det_dataset = pose_det_model.cfg.data['test']['type']

	pose_det_results_list = []
	next_id = 0
	pose_det_results = []
	for frame in mmcv.track_iter_progress(video):
		pose_det_results_last = pose_det_results

		# test a single image, the resulting box is (x1, y1, x2, y2)
		mmdet_results = inference_detector(person_det_model, frame)

		# keep the person class bounding boxes.
		person_det_results = process_mmdet_results(mmdet_results,
												   args.det_cat_id)

		# make person results for single image
		pose_det_results, _ = inference_top_down_pose_model(
			pose_det_model,
			frame,
			person_det_results,
			bbox_thr=args.bbox_thr,
			format='xyxy',
			dataset=pose_det_dataset,
			return_heatmap=False,
			outputs=None)

		# get track id for each person instance
		pose_det_results, next_id = get_track_id(
			pose_det_results,
			pose_det_results_last,
			next_id,
			use_oks=args.use_oks_tracking,
			tracking_thr=args.tracking_thr)

		pose_det_results_list.append(copy.deepcopy(pose_det_results))

	pose_det_results_list[0]
	# Second stage: Pose lifting
	print('Stage 2: 2D-to-3D pose lifting.')

	pose_lift_model = init_pose_model(
		args.pose_lifter_config,
		args.pose_lifter_checkpoint,
		device=args.device.lower())

	assert pose_lift_model.cfg.model.type == 'PoseLifter', \
		'Only "PoseLifter" model is supported for the 2nd stage ' \
		'(2D-to-3D lifting)'
	pose_lift_dataset = pose_lift_model.cfg.data['test']['type']

	if args.out_video_root == '':
		save_out_video = False
	else:
		os.makedirs(args.out_video_root, exist_ok=True)
		save_out_video = True

	if save_out_video:
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		fps = video.fps
		writer = None

	# convert keypoint definition
	for pose_det_results in pose_det_results_list:
		for res in pose_det_results:
			keypoints = res['keypoints']
			res['keypoints'] = convert_keypoint_definition(
				keypoints, pose_det_dataset, pose_lift_dataset)

	# load temporal padding config from model.data_cfg
	if hasattr(pose_lift_model.cfg, 'test_data_cfg'):
		data_cfg = pose_lift_model.cfg.test_data_cfg
	else:
		data_cfg = pose_lift_model.cfg.data_cfg

	# build pose smoother for temporal refinement
	if args.smooth:
		smoother = Smoother(filter_cfg=args.smooth_filter_cfg, keypoint_dim=3)
	else:
		smoother = None

	num_instances = args.num_instances
	pose_lift_results_list = []
	for i, pose_det_results in enumerate(
			mmcv.track_iter_progress(pose_det_results_list)):
		# extract and pad input pose2d sequence
		pose_results_2d = extract_pose_sequence(
			pose_det_results_list,
			frame_idx=i,
			causal=data_cfg.causal,
			seq_len=data_cfg.seq_len,
			step=data_cfg.seq_frame_interval)
		# 2D-to-3D pose lifting
		pose_lift_results = inference_pose_lifter_model(
			pose_lift_model,
			pose_results_2d=pose_results_2d,
			dataset=pose_lift_dataset,
			with_track_id=True,
			image_size=video.resolution,
			norm_pose_2d=args.norm_pose_2d)

		# Pose processing
		pose_lift_results_vis = []
		for idx, res in enumerate(pose_lift_results):
			keypoints_3d = res['keypoints_3d']
			# exchange y,z-axis, and then reverse the direction of x,z-axis
			keypoints_3d = keypoints_3d[..., [0, 2, 1]]
			keypoints_3d[..., 0] = -keypoints_3d[..., 0]
			keypoints_3d[..., 2] = -keypoints_3d[..., 2]
			# rebase height (z-axis)
			if args.rebase_keypoint_height:
				keypoints_3d[..., 2] -= np.min(
					keypoints_3d[..., 2], axis=-1, keepdims=True)
			res['keypoints_3d'] = keypoints_3d
			# add title
			det_res = pose_det_results[idx]
			instance_id = det_res['track_id']
			res['title'] = f'Prediction ({instance_id})'
			# only visualize the target frame
			res['keypoints'] = det_res['keypoints']
			res['bbox'] = det_res['bbox']
			res['track_id'] = instance_id
			pose_lift_results_vis.append(res)

		# Smoothing
		if smoother:
			pose_lift_results = smoother.smooth(pose_lift_results)
			# pose_lift_results[0]["keypoints_3d"].shape

		pose_lift_results_list.append(pose_lift_results)

		# Visualization
		if num_instances < 0:
			num_instances = len(pose_lift_results_vis)
		img_vis = vis_3d_pose_result(
			pose_lift_model,
			result=pose_lift_results_vis,
			img=video[i],
			out_file=None,
			radius=args.radius,
			thickness=args.thickness,
			num_instances=num_instances)


		if save_out_video:
			if writer is None:
				writer = cv2.VideoWriter(
					osp.join(args.out_video_root,
							 f'vis_{osp.basename(args.video_path)}'), fourcc,
					fps, (img_vis.shape[1], img_vis.shape[0]))
			writer.write(img_vis)

	return pose_lift_results_list
	if save_out_video:
		writer.release()


videoPath = "/Users/cameronfranz/Desktop/avatarExperiment1/shibuya2Trim.mov"
# videoPath = "/Users/cameronfranz/Desktop/avatarExperiment1/dancingIrishTrimShort.mp4"
# videoPath = "/Users/cameronfranz/Desktop/avatarExperiment1/shibuyaTrimShort.mp4"
# videoPath = "/Users/cameronfranz/Desktop/avatarExperiment1/shibuyaTrimExtraShort.mp4"
# videoPath = "/Users/cameronfranz/Desktop/avatarExperiment1/mmpose/demo/resources/demo.mp4"
poses = get3dPosesFromVideo(videoPath)
video = mmcv.VideoReader(videoPath)

len(poses[0])

video.fps
plt.rcParams["axes.grid"] = False

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


# input is [17,3] keypoints array w/ hierarchy given by https://mmpose.readthedocs.io/en/latest/api.html#mmpose.datasets.Body3DH36MDataset
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
	# fig.update_layout(
	#     scene = dict(
	#         xaxis = dict(range=[-1.2,1.2],),
	#         yaxis = dict(range=[-1.2,1.2],),
	#         zaxis = dict(range=[-1.2,1.2],),))
	fig.update_layout(scene_aspectmode='data')
	fig.show()

	keypoints3d = pose["keypoints_3d"]

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
		plt.plot([p1[0], p2[0]], [p1[1], p2[1]])

	traverseHierarchy(hierarchy, 0, addToPlot)


for frameIdx in range(0,33, 5):
	poses[frameIdx].sort(key=lambda x: x["track_id"])
	for pose in poses[frameIdx][1:2]:
		keypoints3d = pose["keypoints_3d"].copy()
		keypoints3d[:, 0] = -keypoints3d[:, 0]

		hip = pose["keypoints"][0]
		# plt.scatter(hip[0], hip[1])
		# for (jointId, joint) in enumerate(pose["keypoints"]):
		# 	plt.scatter(joint[0], joint[1], label=jointId, s=0.3, c="red")
		plot2dPose(pose["keypoints"][:, :2])
		# plt.imshow(video[frameIdx])
		# plt.show()

		rightHip3d = pose["keypoints_3d"][1]
		rightKnee3d = pose["keypoints_3d"][2]
		# 3d keypoints have standard length between points
		rightFemur3dto2dProjectionLength = np.linalg.norm(rightHip3d[:2] - rightKnee3d[:2])
		rightHip2d = pose["keypoints"][1]
		rightKnee2d = pose["keypoints"][2]
		rightFemur2dLength = np.linalg.norm(rightHip2d[:2] - rightKnee2d[:2])

		# is scale diff between on the two projections prop to 3d scale?
		scaleFactor = rightFemur2dLength / rightFemur3dto2dProjectionLength
		print(scaleFactor)
		plot3dPose(keypoints3d)

	plt.imshow(video[frameIdx])
	# plt.legend()
	plt.show()

poses[0][0]["keypoints"][0]
poses[10][0]["keypoints"][0]

#------------------------------ UNITY COMMUNICATION ------------------------------

import zmq
import time
import pickle
import plotly.express as px
import quaternion
from collections import OrderedDict

# with open("/tmp/mmposeSave","wb") as f:
#     pickle.dump(pose, f)
# with open("/tmp/mmposeSave","rb") as f:
#     pose = pickle.load(f)


context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.bind('tcp://*:5558')

def toUnityPosition(pos3d):
	newPos = np.zeros(3)
	newPos[0] = 17.7*(pos3d[0]/video.width) + -17.7/2
	newPos[1] = 10*(1-(pos3d[1]/video.height)) + -10/2
	newPos[2] = 10*(pos3d[2]/video.width) + -5
	return newPos

def getPoseQuat(sourceVec, tPoseDir):
	quat = np.zeros(4)
	vec1 = sourceVec
	vec1 = vec1 / np.linalg.norm(vec1)
	vec2 = tPoseDir
	quat[1:4] = np.cross(vec2, vec1);
	quat[0]= (np.linalg.norm(vec1)* np.linalg.norm(vec2)) + np.dot(vec1, vec2)
	quat = quat / np.linalg.norm(quat)
	return quat

def plotVecs(*args):
	xs = []
	ys = []
	zs = []
	colors = []
	for i, point in enumerate(args):
		xs += [0, point[0]]
		ys += [0, point[1]]
		zs += [0, point[2]]
		colors += [str(i), str(i)]
	fig = px.line_3d(x=xs, y=ys, z=zs, color=colors)
	fig.update_layout(
	    scene = dict(
	        xaxis = dict(range=[-1,1],),
	        yaxis = dict(range=[-1,1],),
	        zaxis = dict(range=[-1,1],),))
	# fig.update_layout(scene_aspectmode='cube')
	return fig

# z is up down
def getJointAngles(keypoints3d):
	hierarchy = {1: {2: 3}, 4: {5: 6}, 7: {8: {9: 10, 11: {12: 13}, 14: {15: 16}}}}
	#angle relative. key 1 means bone from 1's parent to 1 -- i.e. bone 0-1
	t_pose_directions = {
		"0-1": np.array([0.4, 0, 0]),
		"1-2": np.array([0, 0, -1.5]),
		"2-3": np.array([0, 0, -1]),

		"0-4": np.array([-0.4, 0, 0]),
		"4-5": np.array([0, 0, -1.5]),
		"5-6": np.array([0, 0, -1]),

		"0-7": np.array([0, 0, 1]),
		"7-8": np.array([0, 0, 1]),
		"8-9": np.array([0, 0, 0.4]),
		"9-10": np.array([0, 0, 0.3]),

		"8-11": np.array([-0.4, 0, 0]),
		"11-12": np.array([-1, 0, 0]),
		"12-13": np.array([-1, 0, 0]),

		"8-14": np.array([0.4, 0, 0]),
		"14-15": np.array([1, 0, 0]),
		"15-16": np.array([1, 0, 0]),
	}

	boneParents = {}
	boneAngles = OrderedDict()
	# only around vertical y-axis, but should also use neck-root to get a fully specified rotation
	rootRotation = getPoseQuat(keypoints3d[1] - keypoints3d[0], [1, 0, 0])
	boneAngles["root"] = rootRotation

	# cumulativeRotations = OrderedDict()
	# cumulativeRotations[0] = np.quaternion(*boneAngles["root"].tolist())
	# plot3dPose(keypoints3d)

	# def getBoneAngles(x, y):
	# 	# y = 5; x=4
	# 	boneId = str(x) + "-" + str(y)
	# 	boneParents[y] = x
	# 	vec1 = keypoints3d[y] - keypoints3d[x]
	# 	vec2 = None
	# 	if x == 0:
	# 		vec2 = t_pose_directions[boneId]
	# 		defaultRot = boneAngles["root"]
	# 	else:
	# 		parentBoneId = str(boneParents[x]) + "-" + str(x)
	# 		vec2 = keypoints3d[x] - keypoints3d[boneParents[x]]
	# 		plotVecs(vec1, vec2)
	# 		# defaultRot is the expected rotation if just in t-pose
	# 		# if reverse direction of x-axis, have to switch order here
	# 		defaultRot = getPoseQuat(t_pose_directions[parentBoneId], t_pose_directions[boneId])
	# 		# defaultRot = getPoseQuat(t_pose_directions[boneId], t_pose_directions[parentBoneId])
	# 	defaultRotQuat = np.quaternion(*defaultRot.tolist())
	# 	vec2 = quaternion.rotate_vectors([defaultRotQuat], vec2)[0]
	# 	# plotVecs(vec1, vec2)
	#
	# 	# vec2 is orientation of bone if rotation hasn't changed from t-pose, vec1 is actual orientation.
	# 	localRotation = getPoseQuat(vec1, vec2)
	# 	applied = quaternion.rotate_vectors([np.quaternion(*localRotation.tolist())], vec2)[0]
	# 	# plotVecs(vec1, vec2, applied)
	# 	# vec2
	# 	# applied = quaternion.rotate_vectors([np.quaternion(*localRotation.tolist())], [-1, 0, 0])[0]
	# 	# plotVecs([-1, 0, 0], applied)
	#
	# 	boneAngles[boneId] = localRotation
	# 	# fullRotation = np.quaternion(*boneAngles[boneId].tolist()) * cumulativeRotations[x]
	# 	# cumulativeRotations[y] = fullRotation

	boneParents = {}
	boneAngles = OrderedDict()
	# only around vertical y-axis, but should also use neck-root to get a fully specified rotation
	# PROBLEM: if two vectors opposite, rotation about any axis will work => hence need neck-root axis
	# hipVec = keypoints3d[1] - keypoints3d[0]
	# rootRotation = getPoseQuat([hipVec[0], 0, 0], [1, 0, 0])
	rootRotation = np.array([0, 0, 0, 1])
	boneAngles["root"] = rootRotation

	# testVec = [0,1,0]
	# applied = quaternion.rotate_vectors([np.quaternion(*rootRotation.tolist())], testVec)[0]
	# plotVecs(testVec, applied)

	def getBoneAngles(x, y):
		boneId = str(x) + "-" + str(y)
		boneParents[y] = x
		vec1 = keypoints3d[y] - keypoints3d[x]
		vec2 = t_pose_directions[boneId]
		# Seems weird to apply do the root rotation like this (seems like will cancel out), but if don't do spine (i.e. a very vertical bone) might be rotated wrong way.
		vec2 = quaternion.rotate_vectors([np.quaternion(*rootRotation.tolist())], vec2)[0]
		rotation = getPoseQuat(vec1, vec2)
		fullRotation = np.quaternion(*rotation.tolist()) * np.quaternion(*rootRotation.tolist())
		boneAngles[boneId] = quaternion.as_float_array(fullRotation)
	traverseHierarchy(hierarchy, 0, getBoneAngles)

	# Test applying the derived bone angles
	testSkeleton = np.zeros((17,3))
	# cumulativeRotations = {}
	# cumulativeRotations[0] = np.quaternion(*boneAngles["root"].tolist())
	def addToSkeleton(x, y):
		# x = 14
		# y = 15
		boneId = str(x) + "-" + str(y)
		dir = t_pose_directions[boneId]
		# fullRotation = np.quaternion(*boneAngles[boneId].tolist()) * cumulativeRotations[x]
		# cumulativeRotations[y] = fullRotation
		fullRotation = np.quaternion(*boneAngles[boneId].tolist()) #* np.quaternion(*rootRotation.tolist())
		# fullRotation = cumulativeRotations[y]
		rotated = quaternion.rotate_vectors([fullRotation], dir)[0]
		# rotations = [
		# 	np.quaternion(*boneAngles["root"].tolist()),
		# 	np.quaternion(*boneAngles[boneId].tolist())
		# ]
		# rotated = dir
		# for r in rotations:
		# 	rotated = quaternion.rotate_vectors([r], rotated)[0]
		# fullRotation
		# plotVecs(dir, rotated)
		testSkeleton[y] = testSkeleton[x] + rotated
	# plot3dPose(keypoints3d)
	# traverseHierarchy(hierarchy, 0, addToSkeleton)
	# # plot3dPose(testSkeleton)

	# plot3dPose(keypoints3d)
	# return cumulativeRotations
	return boneAngles


# spawn character with id 15
message = np.array([1], dtype=np.uint8).tobytes() + np.array([15], dtype=np.int32).tobytes()
socket.send(message)

# for frameIdx in range(18):
for frameIdx in range(0, 34, 1):
	start = time.time()

	# Send image background to Unity
	img = cv2.cvtColor(cv2.flip(video[frameIdx], 0), cv2.COLOR_BGR2RGB).ravel()
	message = np.insert(img, 0, 0)
	socket.send(message.tobytes())

	poses[frameIdx].sort(key=lambda x: x["track_id"])
	for pose in poses[frameIdx][1:2]:
		# plt.scatter(hip[0], hip[1])
		# plt.imshow(video[frameIdx])
		# plt.show()

		#-------------------- SCALING / stand in for z-distance. should use root-neck instead of femur

		hip3d = pose["keypoints_3d"][0]
		spine3d = pose["keypoints_3d"][7]
		spine3dLength = np.linalg.norm(spine3d-hip3d)
		# plot3dPose(keypoints3d)
		spine3dTo2dProjectionLength = np.linalg.norm(hip3d[[0,2]] - spine3d[[0,2]])
		hip2d = 10*pose["keypoints"][0] / video.height
		spine2d = 10*pose["keypoints"][7] / video.height
		spine2dLength = np.linalg.norm(hip2d[:2] - spine2d[:2])
		spine2dLength
		# get target 3d length of spine to send to unity
		targetSpineLength = spine2dLength * (spine3dLength / spine3dTo2dProjectionLength)

		# np.linalg.norm(np.array([0, 0, 0])- np.array([0, 0.079, -0.007])) * 0.56


		# # can get 2d length on screen
		# # np.linalg.norm(1*rightHip3d[:2] - 1*rightKnee3d[:2])
		# # if scale points 2x apart, 3d distance increases by 2x. projection distance also increases 2x
		#
		# # 1. unity femur length is 1.0
		# # 2. using python reference model, can see length 1.0 that projects to size X on screen
		# # 3. target is size Y, so scale the unity model by Y/X
		# = pose["keypoints_3d"][1]
		# rightKnee3d = pose["keypoints_3d"][2]
		# rightFemur3dLength = np.linalg.norm(rightHip3d - rightKnee3d)
		# rightFemur3dto2dProjectionLength = np.linalg.norm(rightHip3d[:2] - rightKnee3d[:2])
		# rightHip2d = 10*pose["keypoints"][1] / video.height
		# rightKnee2d = 10*pose["keypoints"][2] / video.height
		# rightFemur2dLength = np.linalg.norm(rightHip2d[:2] - rightKnee2d[:2])
		# targetFemurLength = rightFemur2dLength * (rightFemur3dLength / rightFemur3dto2dProjectionLength)
		# # print(f"""In current pose, a right femur with length {rightFemur3dLength} projects to a 2d length of {rightFemur3dto2dProjectionLength} on screen. Since the target 2d length on a 10.0x10.0 screen is {rightFemur2dLength}, the target 3d length should be {targetFemurLength}""") #

		# move character with id 15 to pos [0,1,2]
		hip = pose["keypoints"][0]
		hipPosUnity = toUnityPosition(hip)
		hipPosUnity[2] = 0
		hipPosUnity
		message = np.array([2], dtype=np.uint8).tobytes() + np.array([15], dtype=np.int32).tobytes() + hipPosUnity.astype(np.float32).tobytes()
		socket.send(message)

		# scale character with id 15 to have femur length 2.0
		message = np.array([4], dtype=np.uint8).tobytes() + np.array([15], dtype=np.int32).tobytes() +  np.array([targetSpineLength*0.4], dtype=np.float32).tobytes()
		# message = np.array([4], dtype=np.uint8).tobytes() + np.array([15], dtype=np.int32).tobytes() +  np.array([1.8], dtype=np.float32).tobytes()
		socket.send(message)

		#------------------------------ POSE ANGLES V2  / sent as quaternions that rotate from default t-pose

		def quatToUnity(quat):
			swapYZquat = [quat[0], -quat[1], -quat[3], -quat[2]]
			xyzwOrderQuat = [swapYZquat[1], swapYZquat[2], swapYZquat[3], swapYZquat[0]]
			return xyzwOrderQuat

		def quatToUnityTEST(quat):
			swapYZquat = [quat[0], quat[1], -quat[3], quat[2]]
			return swapYZquat


		keypoints3d = pose["keypoints_3d"].copy()
		# plot3dPose(keypoints3d)
		keypoints3d[:, 0] = -keypoints3d[:, 0]
		# Move neck_base keypoint to center of two, because unity kpt doesn't stick out
		keypoints3d[9] = (keypoints3d[8] + keypoints3d[10])/2
		# keypoints3d.shape
		poseAngles = getJointAngles(keypoints3d)
		# poseAngles[15]
		del poseAngles["0-1"]
		del poseAngles["0-4"]
		# poseAngles["root"] = [1,0,0,0]

		anglesOrderedRaw = [item[1] for item in poseAngles.items()] #assumes dict keeps ordering

		# anglesOrderedRaw = [[1, 0, 0, 0] for i in range(15)]
		# anglesOrderedRaw[0] = poseAngles["root"]
		# anglesOrderedRaw[0] = quaternion.as_float_array(poseAngles[0])
		# anglesOrderedRaw[12] = poseAngles["8-14"]
		# anglesOrderedRaw[13] = poseAngles["14-15"]
		# HAVE MISUNDERSTANDING. Because first line below rotates character around z. but second line does not rotate arm around z, even though it would rotate the t-pose arm around z.
		# anglesOrderedRaw[0] = quaternion.as_float_array(quaternion.from_euler_angles([0, 3.14/2, 0]))
		# anglesOrderedRaw[11] = quaternion.as_float_array(quaternion.from_euler_angles([0, 3.14/2, 0]))
		# anglesOrderedRaw[13] = quaternion.as_float_array(quaternion.from_euler_angles([0, 3.14/2, 0]))
		# anglesOrderedRaw[13] = quaternion.as_float_array(poseAngles[15])

		# testVec = [-1, 0, 0]
		# applied = quaternion.rotate_vectors([np.quaternion(*quatToUnityTEST(anglesOrderedRaw[13]))], testVec)[0]
		# applied = quaternion.rotate_vectors([np.quaternion(*anglesOrderedRaw[13])], testVec)[0]
		# plotVecs(testVec, applied)

		# plot3dPose(testSkeleton)
		anglesOrderedUnity = np.array([quatToUnity(item) for item in anglesOrderedRaw]).flatten()
		message = np.array([5], dtype=np.uint8).tobytes() + np.array([15], dtype=np.int32).tobytes() +anglesOrderedUnity.astype(np.float32).tobytes()
		socket.send(message)


		# # ------------------------------ POSE ANGLES / sent as quaternions that rotate from default t-pose
		# # ik with hands, feet, head might be better?
		# # facing away correct if y positive is farther away
		# # facing forward correct if y positive is closer
		# px.scatter_3d(x=pose["keypoints_3d"][:, 0], y=pose["keypoints_3d"][:, 1], z=pose["keypoints_3d"][:, 2])
		#
		#
		# rightHip3d = pose["keypoints_3d"][1]
		# rightKnee3d = pose["keypoints_3d"][2]
		# leftHip3d = pose["keypoints_3d"][4]
		# leftKnee3d = pose["keypoints_3d"][5]
		# # rotation to align the two vectors https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
		#
		# # Here: x, y, z up
		# # Unity: x, y up, z
		# # => must swap last two axes
		# # quaternion lib represents as w,x,y,z, but unity uses x,y,z,w
		# #
		# # def rotCalc:
		# # 	quat = np.zeros(4)
		# # 	vec1 = [femurVec[0], femurVec[2], femurVec[1]]
		# # 	vec1 = vec1 / np.linalg.norm(vec1)
		# # 	vec2 = [tPoseDir[0], tPoseDir[2], tPoseDir[1]]
		# # 	quat[1:4] = np.cross(vec2, vec1);
		# # 	quat[0]= (np.linalg.norm(vec1)* np.linalg.norm(vec2)) + np.dot(vec1, vec2)
		# # 	quat = quat / np.linalg.norm(quat)
		# # 	fullRotation = np.quaternion(*quat.tolist())
		# #
		# # 	leftHipQuat = getPoseQuat(leftKnee3d-leftHip3d, np.array([0, 0, -1]))
		# # 	leftHipQuat
		# # 	fullRotation
		# # 	leftHipQuat
		#
		# leftHipQuatUnity = quatToUnity(getPoseQuat(leftKnee3d-leftHip3d, np.array([0, 0, -1])))
		# rightHipQuatUnity = quatToUnity(getPoseQuat(rightKnee3d-rightHip3d, np.array([0, 0, -1])))
		# # leftHipQuat = [leftHipQuatUnity[3], leftHipQuatUnity[0], leftHipQuatUnity[1], leftHipQuatUnity[2]]
		# # startVec = [0, -1, 0]
		# # rotatedVec = quaternion.rotate_vectors(np.quaternion(*leftHipQuat), startVec)
		# # plotVecs(startVec, rotatedVec)
		#
		# jointAngles = np.array([leftHipQuatUnity, rightHipQuatUnity]).flatten()
		# message = np.array([5], dtype=np.uint8).tobytes() + np.array([15], dtype=np.int32).tobytes() +jointAngles.astype(np.float32).tobytes()
		# socket.send(message)
		#
		# # also need rotation, translation of root joint
		# # def keypoints3dToPoseAngles(keypoints3d):
		# 	# takes in [17x3] float32array of Body3DH36MDataset keypoints.
		# 	# outputs joint angles as angles from t-pose


	end = time.time()
	if (end - start < 1.0/video.fps):
		time.sleep(1.0/video.fps - (end-start))


socket.recv()
socket.send_json(data)
