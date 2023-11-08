import os

import torch
from accelerate import Accelerator, infer_auto_device_map
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, LlamaForCausalLM, AutoTokenizer, LlamaConfig
import deepspeed



# local_rank = int(os.getenv("LOCAL_RANK", "0"))
# world_size = int(os.getenv("WORLD_SIZE", "1"))
# torch.cuda.set_device(local_rank)
# deepspeed.init_distributed()

model_path = r'/root/autodl-fs/model/Phind-CodeLlama-34B-v2'
# accelerator = Accelerator()
# device = accelerator.device
config = LlamaConfig.from_pretrained(model_path)
# train_batch_size = 1 * world_size
# ds_config = {
#     "fp16": {
#         "enabled": False
#     },
#     "bf16": {
#         "enabled": False
#     },
#     "zero_optimization": {
#         "stage": 3,
#         "offload_param": {
#             "device": "cpu",
#             "pin_memory": True
#         },
#         "overlap_comm": True,
#         "contiguous_gradients": True,
#         "reduce_bucket_size": model_hidden_size * model_hidden_size,
#         "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
#         "stage3_param_persistence_threshold": 10 * model_hidden_size
#     },
#     "steps_per_print": 2000,
#     "train_batch_size": train_batch_size,
#     "train_micro_batch_size_per_gpu": 1,
#     "wall_clock_breakdown": False
# }
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
device_map = infer_auto_device_map(model)
print("加载空模型")
print(device_map)

model = load_checkpoint_and_dispatch(model, device_map='auto', checkpoint=model_path, dtype=torch.bfloat16)

model = accelerator.prepare_model(model)
print(model)