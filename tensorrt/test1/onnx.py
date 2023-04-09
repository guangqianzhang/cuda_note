import time

import numpy as np
import onnxruntime 
import onnxruntime 
import torch
import torchvision
import torchvision.models
from torch.utils.data import DataLoader
from tqdm import tqdm

from ppq import*
from ppq .api import*
BATCHSIZE=1

def load_calibration_dataset(use_random_set:bool =False,length:int =256)->Iterable:
    if use_random_set:
        return [torch.rand(size=[BATCHSIZE,3,224,224])  for _ in range(256)]
    # 如果不使用随机数据集，将下载Imagenet
    dataset= torchvision.datasets.ImageFolder(
        root='',
        transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225],inplace=True)
        ])

   )
    dataloader=DataLoader (dataset=dataset,batch_size=BATCHSIZE,shuffle=True)

    #使用海量数据calibration 会很慢，随机从Imagenet中抽取几百张
    load_dataloader=[]
    for data in dataloader:
        load_dataloader.append(data[0])
        if len(load_dataloader)>=256:break
    return load_dataloader


QUANT_PLATFROM=TargetPlatform.PPL_CUDA_INT8  # 量化方式
MODELS={
    'resnet50':torchvision.models.resnet50,
    'mobilenet_v2':torchvision.models.mobilenet_v2,
    'mnas':torchvision.models.mnasnet0_5}
SAMPLES=load_calibration_dataset()
DEVICE='cuda'

for mname,model_builder in MODELS.items():
    print(f'ready for run quantization with {mname}')
    model=model_builder(pretrained=True).to(DEVICE)

    # quantize model with ppq 开始量化
    quantized=quantize_torch_model(
        model=model,calib_dataloader=SAMPLES,collate_fn=lambda x: x.to(DEVICE),
        calib_steps=32,input_shape=[BATCHSIZE,3,224,224],
        setting= QuantizationSettingFactory.default_setting(),
        platform=QUANT_PLATFROM,
        onnx_export_file='model_fp32.onnx'    )
    # 量化后模型跑一边结果，与onnxruntime结果对比
    executor = TorchExecutor(graph=quantized)
    ref_results=[]
    for sample in tqdm(SAMPLES,desc='PPQ GENERATING REFERENCES',total=len(SAMPLES)):
        result=executor.forward(inputs=sample.to(DEVICE))[0]
        result=result.cpu().reshape([BATCHSIZE,1000])
        ref_results.append(result)

    fp32_input_names= [name for name, _ in quantized.inputs.items()]
    fp32_output_names=[name for name,_ in quantized.outputs.items()]

    graphwise_error_analyse(graph=quantized,running_device=DEVICE,dataloader=SAMPLES,
                            collate_fn=lambda x:x.cuda(),steps=32)
    
    # export model to disk
    export_ppq_graph(graph=quantized,
                     platform=TargetPlatform.ONNXRUNTIME,
                     graph_save_to='model_int8.onnx')

    int8_input_names= [name for name, _ in quantized.inputs.items()]
    int8_output_names=[name for name,_ in quantized.outputs.items()]

    # run with onnxruntime
    # 与ppq结果对比
    session=onnxruntime.InferenceSession('model_int8.onnx',providers=['CUDAExecutionProvider'])
    onnxruntime_results=[]
    for sample in tqdm(SAMPLES,desc='ONNXRUNTIME GENERATING OUTPUTS',total=len(SAMPLES)):
        result=session.run([int8_output_names[0]],{int8_input_names[0]:convert_any_to_numpy(sample)})
        result=convert_any_to_torch_tensor(result).reshape([BATCHSIZE,1000])
        onnxruntime_results.append(result)

    # computs sumulationg error
    error=[]
    for ref , real in zip(ref_results,onnxruntime_results):
        error.append(torch_snr_error(ref,real))
    error=sum(error)/len(error)*100
    print(f'PPQ INT8 Simulating Error:{error: .3f}%')

    # benchmark with onnxruntime int8    
    print(f'start Benchmark with onnxruntime (batchsize={BATCHSIZE})')
    benchmark_samples=[np.zeros(shape=[BATCHSIZE,3,224,224],dtype=np.float32) for _ in range(512)]

    session=onnxruntime.InferenceSession('model_fp32.onnx',providers=['CUDAExecutionProvider'])
    tick=time.time()
    for sample in  tqdm(benchmark_samples,desc='FP32 benchmark...'):
        session.run([fp32_output_names[0],{fp32_input_names[0]:sample}])
    tok =time.time()
    print(f'Time span (FP32 MODE):{tok-tick:.4f} sec')

    session=onnxruntime.InferenceSession('model_int8.onnx',providers=['CUDAExecutionProvider'])
    tick=time.time()
    for sample in  tqdm(benchmark_samples,desc='INT8 benchmark...'):
        session.run([int8_output_names[0],{int8_input_names[0]:sample}])
    tok =time.time()
    print(f'Time span (INT8 MODE):{tok-tick:.4f} sec')