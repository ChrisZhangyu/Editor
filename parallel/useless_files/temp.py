from typing import Callable

import colossalai.utils
from colossalai.lazy import LazyInitContext
import torch
import torch.nn as nn

from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin, GeminiPlugin
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device
from colossalai.zero import ColoInitContext, GeminiAdamOptimizer
from datasets import IterableDataset
from torch.optim import *
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM,\
    get_cosine_schedule_with_warmup, AutoConfig
from config import *
from dataset import *
import torch.distributed as dist
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print("PyTorch 版本:", torch.__version__)
os.environ["NCCL_LL_THRESHOLD"] = '0'
os.environ["NCCL_P2P_DISABLE"] = '1'
os.environ["NCCL_IB_DISABLE"] = '1'
os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
os.environ["NCCL_DEBUG"]="INFO"

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['WORLD_SIZE'] = f'{world_size}'
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train(plugin_type: str):
    model_args, training_config = ModelArgs(), TrainConfig()
    setup(0, 1)

    colossalai.launch_from_torch({})

    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path, device="cuda")
    tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path, device="cuda")
    tokenizer.mask_token = "[MASK]"
    tokenizer.sep_token = "[SEP]"
    tokenizer.cls_token = "[CLS]"
    tokenizer.pad_token = "\s"
    dataset = IterableDataset.from_generator(data_iterator,
                                             gen_kwargs={'data_path': model_args.data_path, 'tokenizer': tokenizer})
    train_loader = DataLoader(dataset, collate_fn=collate_gen(tokenizer), batch_size=training_config.batch_size,
                              num_workers=1)
    # from colossalai.utils.
    # train_loader =
    torch.set_default_dtype(torch.bfloat16)
    # 有些属性没设置
    # 存疑，需要试一下
    print("加载模型")
    print(colossalai.utils.get_current_device())
    # with init_empty_weights():
    #     raw_model = LlamaForCausalLM.from_pretrained(model_args.model_name_or_path, config=model_config, torch_dtype=torch.bfloat16)
    with LazyInitContext(default_device=get_current_device()):
        raw_model = LlamaForCausalLM( config=model_config)
    # with ColoInitContext(device=get_current_device()):
    #     raw_model = LlamaForCausalLM(config=model_config)
    # code_llama中没找到bias
    print("准备阶段")

    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in raw_model.named_parameters() if not any(nd for nd in no_decay)],
            "weight_decay": training_config.weight_decay,
        },
        {
            "params": [p for n, p in raw_model.named_parameters() if any(nd for nd in no_decay)],
            "weight_decay": 0,
        }
    ]

    print("加载优化器")
    optimizer = HybridAdam(optimizer_grouped_parameters, lr=training_config.lr, betas=(0.9, 0.95))
    # 为了使所有GPU的学习率与单GPU情况下一致
    # factor = accelerator.num_processes / accelerator.gradient_accumulation_steps
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=training_config.num_warmup_steps * factor,
    #                                             num_training_steps=training_config.num_training_steps * factor)

    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=training_config.num_warmup_steps,
                                                num_training_steps=training_config.num_training_steps)
    print("并行初始化")

    if plugin_type == "HybridParallel":
        print("HybridParallel")

        plugin = HybridParallelPlugin(tp_size=1,
                                      pp_size=2,
                                      microbatch_size=1,
                                      cpu_offload=True,
                                      enable_all_optimization=True,
                                      zero_stage=1,
                                      precision='bf16',
                                      initial_scale=1)
    elif plugin_type == "gemini":
        print("gemini")

        plugin = GeminiPlugin(
            precision="bf16",
            initial_scale=2**16,
            verbose=True,
            search_range_m=128,
            growth_factor=1,
            hidden_dim=model_config.hidden_size
        )
    else:
        raise RuntimeError("不支持的插件类型")

    booster = Booster(plugin=plugin)
    dist.get_rank()
    model, optimizer, criterion,  train_dataloader, scheduler = booster.boost(raw_model, optimizer, nn.CrossEntropyLoss
, train_loader, scheduler)
    booster.load_model(model, model_args.model_name_or_path)
    torch.cuda.synchronize()


    for epoch in range(10):
        one_epoch(epoch, model, optimizer, criterion, scheduler, train_dataloader, booster)


def one_epoch(epoch,
              model: nn.Module,
              optimizer: Optimizer,
              criterion: Callable,
              scheduler: LRScheduler,
              train_dataloader: DataLoader,
              booster: Booster):
    # print(type(train_dataloader))
    # total_step = len(train_dataloader)
    total_step = 1000
    train_loader_iter = iter(train_dataloader)
    print("开始训练")
    model.train()
    optimizer.zero_grad()
    with tqdm(range(total_step),
              desc=f'Epoch [{epoch + 1}]') as pbar:
        for _ in pbar:
            batch = next(train_loader_iter)
            # print(f"model_device: {model.device}")
            for k, v in batch.items():
                batch[k] = v.to(torch.cuda.current_device())
            outputs = model(**batch)
            loss = outputs[0]
            print(loss.size())

            booster.backward(loss, optimizer)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()



if __name__ == '__main__':
   train("gemini")
