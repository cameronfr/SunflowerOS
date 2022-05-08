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

import mediapipe as mp

import sys
sys.path.append("/Users/cameronfranz/Desktop/avatarExperiment1/Resources/mmdetection/")
sys.path.append("/Users/cameronfranz/Desktop/avatarExperiment1/Resources/mmdetection3d/")
sys.path.append("/Users/cameronfranz/Desktop/avatarExperiment1/Resources/mmpose/")

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
class DictObj:
	def __init__(self, in_dict:dict):
		assert isinstance(in_dict, dict)
		for key, val in in_dict.items():
			if isinstance(val, (list, tuple)):
			   setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
			else:
			   setattr(self, key, DictObj(val) if isinstance(val, dict) else val)

def get3dPosesFromVideoMMPose(videoPath, numFrames=-1):
	assert has_mmdet, 'Please install mmdet to run the demo.'

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
		person_det_results = process_mmdet_results(mmdet_results, cat_id=1)

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
		pose_det_results

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

def get3dPosesFromVideoMediapipe(videoPath):
	video = mmcv.VideoReader(videoPath)
	assert video.opened, f'Failed to load video file {args.video_path}'
	# Extract bounding boxes
	os.chdir("/Users/cameronfranz/Desktop/avatarExperiment1/Resources/mmpose")
	args = DictObj({
		"det_config": "demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py",
		"det_checkpoint": "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
		"device": "CPU",
		"use_oks_tracking": True,
		"tracking_thr": 0.3,
		"smooth_filter_cfg": "configs/_base_/filters/one_euro.py",
	})
	smoother = Smoother(filter_cfg=args.smooth_filter_cfg, keypoint_dim=3)
	person_det_model = init_detector(args.det_config, args.det_checkpoint, device=args.device.lower())
	pose_model = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.51)

	next_id = 0
	pose_det_results_tracked_last = []
	pose_det_results_list = []
	for idx, frame in enumerate(mmcv.track_iter_progress(video)):
		# if (idx == 4):
		# 	break
		# frame = video[0]
		# pose_det_results_last = pose_det_results

		# test a single image, the resulting box is (x1, y1, x2, y2)
		mmdet_results = inference_detector(person_det_model, frame)
		# keep the person class bounding boxes.
		person_det_results = process_mmdet_results(mmdet_results, cat_id=1)

		pose_det_results = []
		for personDetection in person_det_results:
			bbox = personDetection["bbox"]
			confidence = bbox[4]
			if confidence < 0.9:
				continue
			(x1, y1, x2, y2) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
			personImageBGR = frame[y1:y2, x1:x2] #PIL images are row,col i.e. height,width or y,x
			personImageRGB = cv2.cvtColor(personImageBGR, cv2.COLOR_BGR2RGB)
			pose_model_result = pose_model.process(personImageRGB)
			if not pose_model_result.pose_world_landmarks:
				# print("Failed to extract pose")
				# plt.imshow(personImageRGB)
				# plt.show()
				continue
			pose_array_mpformat_world = np.empty((33, 3)) # not in image coords
			pose_array_mpformat_2d = np.empty((33, 3)) # in image coords
			for i in range(33):
				pose_array_mpformat_world[i, 0] = pose_model_result.pose_world_landmarks.landmark[i].x
				pose_array_mpformat_world[i, 1] = pose_model_result.pose_world_landmarks.landmark[i].y
				pose_array_mpformat_world[i, 2] = pose_model_result.pose_world_landmarks.landmark[i].z
				pose_array_mpformat_2d[i, 0] = pose_model_result.pose_landmarks.landmark[i].x
				pose_array_mpformat_2d[i, 1] = pose_model_result.pose_landmarks.landmark[i].y
				pose_array_mpformat_2d[i, 2] = pose_model_result.pose_landmarks.landmark[i].z
			def keypointsMPtoB36M(keypointsMP, is3d=False):
				keypointsB36M = np.empty((17, 3))
				# hips
				keypointsB36M[0] = np.average(keypointsMP[[23, 24]], axis=0)
				# legs
				keypointsB36M[[1,2,3,4,5,6]] = keypointsMP[[24, 26, 28, 23, 25, 27]]
				# spine and thorax and neck
				keypointsB36M[7] = np.average(keypointsMP[[11, 12, 23, 24]], axis=0)
				keypointsB36M[8] = np.average(keypointsMP[[11, 12]], axis=0)
				keypointsB36M[9] = np.average(keypointsMP[[9, 10]], axis=0)

				# est head keypoint by moving back into head from nose in correct dir
				if (is3d):
					faceWidthVector = keypointsMP[7] - keypointsMP[8]
					faceHeightVector = np.average(keypointsMP[[9, 10]], axis=0) - np.average(keypointsMP[[3, 6]], axis=0)
					faceCross = np.cross(faceWidthVector, faceHeightVector)
					faceCross = faceCross / np.linalg.norm(faceCross)
					keypointsB36M[10] = keypointsMP[0] + faceCross * 0.097 *3
					# print(faceWidthVector, faceHeightVector, keypointsMP[0], faceCross, keypointsB36M[10])
				else:
					keypointsB36M[10] = keypointsMP[0]
				# arms
				keypointsB36M[[11, 12, 13, 14, 15, 16]] = keypointsMP[[11, 13, 15, 12, 14, 16]]
				return keypointsB36M
			pose_array_b36mformat_world = keypointsMPtoB36M(pose_array_mpformat_world, is3d=True)
			pose_array_b36mformat_2d = keypointsMPtoB36M(pose_array_mpformat_2d)

			keypoints2d = np.empty((17, 3))
			keypoints2d[:, 0] = bbox[0] + pose_array_b36mformat_2d[:, 0] * personImageBGR.shape[1]
			keypoints2d[:, 1] = bbox[1] + pose_array_b36mformat_2d[:, 1] * personImageBGR.shape[0]
			keypoints2d[:, 2] = 0.95 #fake confidence score since model does not output them

			keypoints3d = np.empty((17, 3))
			# z positive is up. Also reverse x (idk why mmpose does that in orig fn)
			keypoints3d[..., 0] = -pose_array_b36mformat_world[..., 0]
			keypoints3d[..., 1] = pose_array_b36mformat_world[..., 2]
			keypoints3d[..., 2] = -pose_array_b36mformat_world[..., 1]
			# plot3dPose(keypoints3d)

			# plt.imshow(frame)
			# for k in keypoints2d:
			# 	plt.scatter(k[0], k[1])
			# plt.show()
			# plt.imshow(frame)
			# for k in keypoints3d:
			# 	plt.scatter(k[0], k[2])
			# plt.xlim(-1, 1)
			# plt.ylim(-1, 1)
			# plt.show()

			pose_det_results.append({
				"bbox": bbox,
				"keypoints": keypoints2d,
				"keypoints_3d": keypoints3d,
			})

		#get_track_id expects the "keypoints" attr to be [x,y,prob]
		pose_det_results_tracked, next_id = get_track_id(
			pose_det_results,
			pose_det_results_tracked_last,
			next_id,
			use_oks=args.use_oks_tracking,
			tracking_thr=args.tracking_thr)
		pose_det_results_tracked_last = pose_det_results_tracked
		# pose_det_results_tracked_last[0]["keypoints"].shape
		# pose_det_results[0]["keypoints"].reshape((-1))
		# pose_det_results

		pose_det_results_tracked_smooth = smoother.smooth(pose_det_results_tracked)
		pose_det_results_list.append(copy.deepcopy(pose_det_results_tracked_smooth))
	poses = pose_det_results_list
	return poses

# videoPath = "/Users/cameronfranz/Desktop/avatarExperiment1/shibuya2Trim.mov"
# videoPath = "/Users/cameronfranz/Desktop/avatarExperiment1/Resources/shibuyaTrim1.mp4"
videoPath = "/Users/cameronfranz/Desktop/avatarExperiment1/Resources/dancingIrishTrim.mp4"
# videoPath = "/Users/cameronfranz/Desktop/avatarExperiment1/shibuyaTrimShort.mp4"
# videoPath = "/Users/cameronfranz/Desktop/avatarExperiment1/Resources/shibuyaTrimExtraShort.mp4"
# videoPath = "/Users/cameronfranz/Desktop/avatarExperiment1/mmpose/demo/resources/demo.mp4"
# poses = get3dPosesFromVideoMMPose(videoPath)
poses = get3dPosesFromVideoMediapipe(videoPath)
video = mmcv.VideoReader(videoPath)

poses.shape
#------------------------------

plt.rcParams["axes.grid"] = False
#------------------------------

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

#------------------------------ UNITY COMMUNICATION ------------------------------

# frame = video[0]
# pose = poses[0][0]
#
# plt.imshow(frame)
# for k in pose["keypoints"]:
# 	plt.scatter(k[0], k[1])
# plt.show()
# plot3dPose(pose["keypoints_3d"])
# poses[0][0]["keypoints_3d"]
# poses[0]

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

	boneParents = {}
	boneAngles = OrderedDict()
	# only around vertical y-axis, but should also use neck-root to get a fully specified rotation
	# PROBLEM: if two vectors opposite, rotation about any axis will work => hence need neck-root axis

	# plot3dPose(keypoints3d)
	hipVec = keypoints3d[1] - keypoints3d[0]
	spineVec = keypoints3d[7] - keypoints3d[0]
	bodyCross = np.cross(spineVec, hipVec)
	basisChange = np.empty((3, 3))
	basisChange[:, 0] = hipVec/np.linalg.norm(hipVec)
	basisChange[:, 1] = bodyCross/np.linalg.norm(bodyCross)
	basisChange[:, 2] = spineVec/np.linalg.norm(spineVec)
	rootRotation = quaternion.from_rotation_matrix(basisChange, nonorthogonal=True)
	rootRotation = quaternion.as_float_array(rootRotation)
	# rootRotation = np.array([0, 0, 0, 1])
	boneAngles["root"] = rootRotation

	# plotVecs(hipVec*10, [1,0,0])
	# testVec = [1,0,0]
	# applied = basisChange @ testVec
	# applied = quaternion.rotate_vectors([np.quaternion(*rootRotation.tolist())], testVec)[0]
	# plotVecs(testVec, applied)

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
		boneId = str(x) + "-" + str(y)
		dir = t_pose_directions[boneId]
		fullRotation = np.quaternion(*boneAngles[boneId].tolist()) #* np.quaternion(*rootRotation.tolist())
		rotated = quaternion.rotate_vectors([fullRotation], dir)[0]
		testSkeleton[y] = testSkeleton[x] + rotated
	# plot3dPose(keypoints3d)
	# traverseHierarchy(hierarchy, 0, addToSkeleton)
	# # plot3dPose(testSkeleton)

	return boneAngles


frame = video[0]
results.segmentation_mask.shape
frame.shape
multeed = frame * results.segmentation_mask[:, :, np.newaxis]
plt.imshow(multeed.astype(np.uint8))

humanSegmenterModel = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
# spawn character with id 15
# message = np.array([1], dtype=np.uint8).tobytes() + np.array([15], dtype=np.int32).tobytes()
# socket.send(message)
avatarWithIdExists = {}
for frameIdx in range(0, 1209, 1):
	start = time.time()

	# Send image background to Unity
	segmentResults = humanSegmenterModel.process(cv2.cvtColor(video[frameIdx], cv2.COLOR_BGR2RGB))
	blurred = cv2.blur(video[frameIdx], (100, 100))
	masked = (video[frameIdx] * (1-segmentResults.segmentation_mask[:, :, np.newaxis])) + (blurred * segmentResults.segmentation_mask[:, :, np.newaxis] * 1.5)
	masked = np.clip(masked,0,255).astype(np.uint8)
	# img = cv2.cvtColor(cv2.flip(video[frameIdx], 0), cv2.COLOR_BGR2RGB).ravel()
	img = cv2.cvtColor(cv2.flip(masked, 0), cv2.COLOR_BGR2RGB).ravel()
	message = np.insert(img, 0, 0)
	socket.send(message.tobytes())

	# Despawn Objects no longer in scene
	existingAvatarIds = set(avatarWithIdExists.keys())
	idsInScene = []
	for pose in poses[frameIdx]:
		idsInScene.append(pose["track_id"])
	idsToDespawn = existingAvatarIds.difference(idsInScene)
	for id in idsToDespawn:
		message = np.array([3], dtype=np.uint8).tobytes() + np.array([id], dtype=np.int32).tobytes()
		socket.send(message)
		del avatarWithIdExists[id]


	for pose in poses[frameIdx]:
		# Spawn in new objects
		trackId = pose["track_id"]
		if trackId not in avatarWithIdExists:
			message = np.array([1], dtype=np.uint8).tobytes() + np.array([trackId], dtype=np.int32).tobytes()
			socket.send(message)
			avatarWithIdExists[trackId] = True

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

		hip = [*pose["keypoints"][0].tolist(), 1] # jankily add another point cause toUnityPosition needs list of len >=3
		hipPosUnity = toUnityPosition(hip)
		hipPosUnity[2] = 0 # zero out, since hip[3] is accuracy probability
		# hipPosUnity[1] += 0.5
		# hipPosUnity[0] += 2.5
		message = np.array([2], dtype=np.uint8).tobytes() + np.array([trackId], dtype=np.int32).tobytes() + hipPosUnity.astype(np.float32).tobytes()
		socket.send(message)

		# scale character with id 15 to have femur length 2.0
		message = np.array([4], dtype=np.uint8).tobytes() + np.array([trackId], dtype=np.int32).tobytes() +  np.array([targetSpineLength*0.35], dtype=np.float32).tobytes()
		# message = np.array([4], dtype=np.uint8).tobytes() + np.array([15], dtype=np.int32).tobytes() +  np.array([1.8], dtype=np.float32).tobytes()
		socket.send(message)

		#------------------------------ POSE ANGLES V2  / sent as quaternions that rotate from default t-pose

		def quatToUnity(quat):
			swapYZquat = [quat[0], -quat[1], -quat[3], -quat[2]]
			xyzwOrderQuat = [swapYZquat[1], swapYZquat[2], swapYZquat[3], swapYZquat[0]]
			return xyzwOrderQuat

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

		# applied = quaternion.rotate_vectors([np.quaternion(*anglesOrderedRaw[13])], testVec)[0]
		# plotVecs(testVec, applied)

		# plot3dPose(testSkeleton)
		anglesOrderedUnity = np.array([quatToUnity(item) for item in anglesOrderedRaw]).flatten()
		message = np.array([5], dtype=np.uint8).tobytes() + np.array([trackId], dtype=np.int32).tobytes() +anglesOrderedUnity.astype(np.float32).tobytes()
		socket.send(message)

	end = time.time()
	if (end - start < 1.0/video.fps):
		time.sleep(1.0/video.fps - (end-start))


socket.recv()
socket.send_json(data)
