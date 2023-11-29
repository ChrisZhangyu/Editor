import json
import os
import time
from typing import Callable, Union

import accelerate
import datasets
from accelerate import infer_auto_device_map
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate import init_empty_weights
from datasets import IterableDataset
from deepspeed.compression.helper import recursive_getattr, recursive_setattr
from torch.optim import *
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, \
    get_cosine_schedule_with_warmup, AutoConfig

from Editor.parallel.lora_layers import LinearLayer_LoRA
from config import *
from torch.utils.tensorboard import SummaryWriter
from dataset import *
import torch.distributed as dist
from deepspeed import get_accelerator, init_distributed

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print("PyTorch 版本:", torch.__version__)
os.environ["NCCL_LL_THRESHOLD"] = '0'
os.environ["NCCL_P2P_DISABLE"] = '1'
os.environ["NCCL_IB_DISABLE"] = '1'
os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
os.environ["NCCL_DEBUG"] = "INFO"

inference_save_path = "/root/autodl-tmp/Editor/parallel/results/raw_tora_13B"
checkpoint_save_path = "/root/autodl-tmp/Editor/parallel/"


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def get_train_ds_config(offload,
                        dtype,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=1024,
                        enable_tensorboard=False,
                        enable_mixed_precision_lora=False,
                        tb_path="",
                        tb_name="",
                        global_batch_size=32,
                        micro_batch_size=4):

    device = "cpu" if offload else "none"
    if dtype == "fp16":
        data_type = "fp16"
        dtype_config = {"enabled": True, "loss_scale_window": 100}
    elif dtype == "bf16":
        data_type = "bfloat16"
        dtype_config = {"enabled": True}
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }
    if enable_mixed_precision_lora:
        # 将冻结的参数进行量化保存，提高效率
        zero_opt_dict["zero_quantized_nontrainable_weights"] = True
        if dist.get_world_size() != get_accelerator().device_count():
            zero_opt_dict["zero_hpz_partition_size"] = get_accelerator(
            ).device_count()
    return {
        "train_batch_size": global_batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "data_type": dtype_config,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard"
        }
    }


def wrap_with_lora_linear(model: torch.nn.Module, part_module_name: list, lora_rank, lora_scaling=1, lora_dropout=0):
    # 统计哪些层需要用lora替换
    replace_layers = []
    for name, module in model.named_modules():
        # print(f"name:{name}\tmodule:{module}")
        if isinstance(module, nn.Linear) and any([item in name for item in part_module_name]):
            replace_layers.append(name)
    for layer in replace_layers:
        module = recursive_getattr(model, layer)
        lora_linear = LinearLayer_LoRA(
                            module.weight,
                            lora_rank,
                            lora_scaling,
                            lora_dropout,
                            module.bias).to(module.weight.device).to(module.weight.dtype)
        recursive_setattr(model, layer, lora_linear)
    return model


def freeze_main_model(model, optimize_params=Union[list, None]):
    for name, param in model.named_parameters():
        if "lora_right_weight" in name or "lora_left_weight" in name or name in optimize_params:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def lora_train(is_lora=True):
    model_args, training_config = ModelArgs(), TrainConfig()
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
    model = accelerate.load_checkpoint_and_dispatch(model, model_args.model_name_or_path,device_map="auto")
    model = wrap_with_lora_linear(model, ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"], lora_rank=8)
    model = freeze_main_model(model)
    print(model)




if __name__ == '__main__':
    # inference("HumanEval", "")
    lora_train()
    # merge_json()
