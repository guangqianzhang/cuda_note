from   typing import  Iterable
import torch
import torchvision

from ppq import *
from ppq.api import *

INPUT_SHAPE=[32,3,224,224]
DEVICE='cuda'
PLATFORM= TargetPlatform.PPL_CUDA_INT8

def load_calibration_dataset()->Iterable:
    return [torch.rand(INPUT_SHAPE) for _ in range(32)]

def collate_fn(batch:torch.Tensor)->torch.Tensor:
    return batch.to(DEVICE)

model=torchvision.models.shufflenet_v2_x1_0(pretrained=True)
model=model.to(DEVICE)

# 量化配置信息
setting =QuantizationSettingFactory.pplcuda_setting()
ir = quantize_torch_model(
    model=model,calib_dataloader=load_calibration_dataset(),setting=setting,
    calib_steps=8,input_shape=INPUT_SHAPE,collate_fn=collate_fn,
    platform = PLATFORM)

# 量化误差分析

reports=layerwise_error_analyse(
    graph=ir,running_device=DEVICE,collate_fn=collate_fn,
    dataloader=load_calibration_dataset())

reports= graphwise_error_analyse(
    graph=ir,running_device=DEVICE,collate_fn=collate_fn,
    dataloader=load_calibration_dataset())

# 导出模型 最后
export_ppq_graph(graph=ir,platform=TargetPlatform.ONNXRUNTIME,graph_save_to='quantized_shufflenet.onnx')
