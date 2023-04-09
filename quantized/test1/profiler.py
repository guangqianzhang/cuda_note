"""
可通过tensorboard参看CPU GPU 各算子 执行时间 
发现真正在执行延迟上占主导地位的部分进行针对性优化
GPU的问题:优化网络
CPU问题:如何把数据送进GPU
"""
import torch.profiler
from tqdm import tqdm 
from ppq import TorchExecutor, TargetPlatform
from ppq.api import *
sample_input=[torch.rand(128,3,224,224) for i in range(32)]
ir =quantize_onnx_model(onnx_import_file='working/resnet18-v1-7.onnx',
                        calib_dataloader=sample_input,
                        calib_steps=16,
                        do_quantize=False,
                        input_shape=None,
                        collate_fn=lambda x: x.to('cuda'),
                        platform = TargetPlatform.TRT_INT8,
                        inputs=torch.rand(1,3,244,244).to('cuda')
                        )

executor=TorchExecutor(ir)
with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=2,warmup=2,active=6,repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        dir_name='working/performance'    ),
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
                ],
    with_stack=True,
) as profiler:
    with torch.no_grad():
        for batch_idx in tqdm(range(16),desc='profiling....'):
            executor.forward(sample_input[0].to('cuda'))
            profiler.step()
