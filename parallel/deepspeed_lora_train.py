import os
import time
from typing import Callable, Union

import accelerate
import datasets
import deepspeed
from accelerate import infer_auto_device_map
import torch
from deepspeed.ops.adam import FusedAdam
import torch.nn as nn
from accelerate import init_empty_weights
from datasets import IterableDataset
from deepspeed.compression.helper import recursive_getattr, recursive_setattr
from torch.optim import *
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, \
    get_cosine_schedule_with_warmup, AutoConfig, get_scheduler

from lora_layers import LinearLayer_LoRA
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


def get_train_ds_config(dtype='bf16',
                        stage=3,
                        offload=False,
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
                        global_batch_size=2,
                        micro_batch_size=1):
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
    result_dict = {
        "train_batch_size": 2,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        data_type: dtype_config,
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
        "gradient_accumulation_steps": 1,
        "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard"
        }
    }
    return result_dict


def setup(rank, world_size=1):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


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


def freeze_main_model(model, optimize_params: Optional[list] = []):
    for name, param in model.named_parameters():
        if "lora_right_weight" in name or "lora_left_weight" in name or name in optimize_params:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


# 将input_embeddings的梯度设置为True，来兼容gradient_checkpointing。(没有完全理解这一步的操作)
def make_model_gradient_checkpointing_compatible(model):
    # Higgingface added this enable input require grads function to make gradient checkpointing work for lora-only optimization
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model, "get_input_embeddings"):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(
            make_inputs_require_grad)
    return model


def prepare_model_for_lora(model: torch.nn.Module, lora_rank):
    model = wrap_with_lora_linear(model,
                                  ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
                                  lora_rank=lora_rank)
    model = freeze_main_model(model)
    model = make_model_gradient_checkpointing_compatible(model)
    return model


def prepare_params_for_optimizer(model: nn.Module, weight_decay, lora_lr, no_decay_name_list=None, lora_name_list=None):
    decay = ["bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"]
    lora_param = []
    if no_decay_name_list:
        decay = no_decay_name_list
    if lora_name_list:
        lora_param = lora_param
    # 选出非decay的所有参数
    params_to_optimize = [{
        'params': [p for n, p in model.named_parameters()
                   if (not any(nd in n.lower() for nd in decay)) and p.requires_grad and not any(
                l in n.lower() for l in lora_param)],
        'weight_decay': weight_decay
    },
        {
            'params': [p for n, p in model.named_parameters()
                       if (not any(nd in n.lower() for nd in decay)) and p.requires_grad and any(
                    l in n.lower() for l in lora_param)],
            'weight_decay': weight_decay,
            'lr': lora_lr

        },
        {
            'params': [p for n, p in model.named_parameters()
                       if (any(nd in n.lower() for nd in decay)) and p.requires_grad],
            'weight_decay': 0

        }]

    non_empty_groups = []
    for group in params_to_optimize:
        if group["params"]:
            non_empty_groups.append(group)
    return non_empty_groups




def lora_train(local_rank=1, is_lora=True):
    # setup(local_rank, 1)
    model_args, training_config = ModelArgs(), TrainConfig()
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    torch.set_default_dtype(torch.bfloat16)
    tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path, device=training_config.device)

    assert local_rank >= -1
    accelerator = get_accelerator()
    if local_rank == -1:
        device = torch.device(accelerator.device_name())
    else:
        accelerator.set_device(local_rank)
        device = torch.device(accelerator.device_name(), local_rank)
        deepspeed.init_distributed()

    dataset = IterableDataset.from_generator(data_iterator, gen_kwargs={'data_or_path': model_args.data_path,
                                                                        'tokenizer': tokenizer,
                                                                        'mode': 'train',
                                                                        'dataset_type': 'API'})
    print(device)
    # train_sampler = RandomSampler(dataset)

    train_loader = DataLoader(dataset,
                              collate_fn=collate_gen(tokenizer, training_config.max_length),
                              # sampler=train_sampler,
                              batch_size=training_config.batch_size,
                              num_workers=1)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
    model = accelerate.load_checkpoint_and_dispatch(model, model_args.model_name_or_path, device_map="auto", dtype=torch.bfloat16)
    model = prepare_model_for_lora(model, 8)

    optimizer = FusedAdam(prepare_params_for_optimizer(model, training_config.weight_decay, training_config.lora_lr),
                          lr=training_config.lr, betas=(0.9, 0.95))

    scheduler = get_scheduler(name=training_config.lr_scheduler_type,
                              optimizer=optimizer,
                              num_training_steps=training_config.num_training_steps,
                              num_warmup_steps=training_config.num_warmup_steps)
    ds_config = get_train_ds_config(global_batch_size=training_config.batch_size,
                                    micro_batch_size=training_config.batch_size
                                    )
    model, optimizer, _, lr_scheduler = deepspeed.initialize(model=model,
                                                             optimizer=optimizer,
                                                             config=ds_config,
                                                             lr_scheduler=scheduler,
                                                             )
    count = 0
    for step, batch in enumerate(train_loader):
        start = time.time()
        count += 1
        print(f"{accelerator.current_device()}: {count}")
        # for k, v in batch.items():
        #     batch[k] = v.to(device)
        # output = model(**batch, use_cache=False)
        # loss = output.loss
        # print(loss)
        # model.backward(loss)
        # model.step()


if __name__ == '__main__':
    # inference("HumanEval", "")
    lora_train()
    # merge_json()
