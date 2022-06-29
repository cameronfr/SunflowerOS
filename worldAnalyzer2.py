# Updated version. Simplified.

import torch
import torchvision
import numpy as np
import mmdet.apis
import os
import av


import matplotlib.pyplot as plt
# plt.style.use("ggplot")
%config InlineBackend.figure_format='svg'


os.chdir("/home/cameron/mmdetection/")

device = torch.cuda.current_device()
modelConfig = "configs/yolox/yolox_x_8x8_300e_coco.py"
modelCheckpoint = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth"
model = mmdet.apis.init_detector(modelConfig, modelCheckpoint, device) # this is a torch module.
sum([x.numel() for x in model.parameters()])

videoRaw = av.open("/home/cameron/shibuyaTrim1.mp4")
videoRaw.duration # in microseconds
videoRaw.seek(3*10**6)
videoFrames = videoRaw.decode(videoRaw.streams[0])

frame = np.array(next(videoFrames).to_image())
frames = frame[np.newaxis, ...] # B, H, W, RGB -- this is format for passing images

# DIY PIPELINE
def processFrames(frames):
    imageTransforms = torchvision.transforms.Compose([
        torchvision.transforms.Pad((0, 0, 0, 560), fill=114),
        torchvision.transforms.Resize((640, 640)),
    ])
    processed = torch.tensor(frames)[:, :, :, (2, 1, 0)] # BGR -> RGB
    processed = processed.permute(0, 3, 1, 2) # (Batch, W, H, C) -> (Batch, C, W, H)
    processed.shape
    processed = imageTransforms(processed)
    return processed
processedFrames = processFrames(frames).repeat((2, 1, 1, 1))

# CONFIG PIPELINE AND INFERENCE
import mmcv
from mmdet.datasets.pipelines import Compose
from mmdet.datasets import replace_ImageToTensor
config = mmcv.Config.fromfile(modelConfig)
config.data.test.pipeline[0].type = 'LoadImageFromWebcam'
config.data.test.pipeline = replace_ImageToTensor(config.data.test.pipeline)
test_pipeline = Compose(config.data.test.pipeline)
pipelineInput = dict(img=frames[0, :, :, ::-1])
pipelineOutput = test_pipeline(pipelineInput)
# plt.imshow(pipelineOutput["img"][0].data.numpy().transpose(1, 2, 0).astype(np.uint8))
img = pipelineOutput["img"][0].data[None, :, :, :].to(device)
img_metas = pipelineOutput["img_metas"][0].data
with torch.no_grad():
    out = model.forward_test([img], [[img_metas]],rescale=True)
resVis = model.show_result(frames[0, :, :, ::1], out[0], score_thr=0.8, bbox_color=None, text_color=(200, 200, 200), mask_color=None)
plt.imshow(resVis)

# REFERENCE INFERENCE
outRef = mmdet.apis.inference_detector(model, frames[0, :, :, ::-1])
resVis = model.show_result(frames[0, :, :, ::1], outRef, score_thr=0.8, bbox_color=None, text_color=(200, 200, 200), mask_color=None)
plt.imshow(resVis)


# INFERENCE RECONSTRUCTED
# plt.imshow(processedFrames[0].permute(1, 2, 0))
with torch.no_grad():
    img_meta = {"img_shape": (640, 640), "flip": False, "scale_factor":1}
    input = processedFrames.type(torch.FloatTensor).to(device)
    img_metas = [img_meta]*input.shape[0]
    input.shape
    out = model.forward_test([input], [img_metas], rescale=False)
resVis = model.show_result(processedFrames[0].numpy().transpose(1, 2, 0), out[0], score_thr=0.8, bbox_color=None, text_color=(200, 200, 200), mask_color=None)
plt.imshow(resVis)
