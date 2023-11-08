import json
import os
from typing import Callable

import accelerate
import colossalai.utils
import datasets
from accelerate import infer_auto_device_map
from colossalai.inference.tensor_parallel import TPInferEngine
from colossalai.lazy import LazyInitContext
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate import init_empty_weights
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
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train(plugin_type: str):
    model_args, training_config = ModelArgs(), TrainConfig()
    colossalai.launch_from_torch({})
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path, device="cuda")
    tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path, device="cuda")
    tokenizer.mask_token = "[MASK]"
    tokenizer.sep_token = "[SEP]"
    tokenizer.cls_token = "[CLS]"
    tokenizer.pad_token = "\s"
    dataset = IterableDataset.from_generator(data_iterator, gen_kwargs={'data_path': model_args.data_path, 'tokenizer': tokenizer})
    train_loader = DataLoader(dataset,
                              collate_fn=collate_gen(tokenizer, training_config.max_length),
                              batch_size=training_config.batch_size,
                              num_workers=1)

    torch.set_default_dtype(torch.bfloat16)
    # 有些属性没设置
    # 存疑，需要试一下
    print("加载模型")
    print(colossalai.utils.get_current_device())

    with LazyInitContext(default_device=get_current_device()):
        raw_model = LlamaForCausalLM( config=model_config)

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
            placement_policy="auto",
            precision="bf16",
            initial_scale=2**16,
            verbose=True,
            search_range_m=128,
            # offload_optim_frac=0.2,
            growth_factor=1,
            hidden_dim=model_config.hidden_size,
        )
    else:
        raise RuntimeError("不支持的插件类型")

    booster = Booster(plugin=plugin, mixed_precision="")
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
            for k, v in batch.items(dist.get_rank()):
                batch[k] = v.to(torch.cuda.current_device())

            outputs = model(**batch)
            loss = outputs[0]
            booster.backward(loss, optimizer)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

def inference(dataset_type):
    model_args, training_config = ModelArgs(), TrainConfig()
    # colossalai.launch_from_torch({})
    setup(0, 1)
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path, device="cuda")
    tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path, device="cuda")
    tokenizer.mask_token = "[MASK]"
    tokenizer.sep_token = "[SEP]"
    tokenizer.cls_token = "[CLS]"
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"
    dataset = IterableDataset.from_generator(data_iterator,
                                             gen_kwargs={'data_or_path': model_args.data_path, 'tokenizer': tokenizer, 'dataset_type': dataset_type, 'mode': 'train', 'max_length': 1024})
    test_loader = DataLoader(dataset, collate_fn=collate_gen(tokenizer, "infer"), batch_size=training_config.batch_size,
                              num_workers=1)
    torch.set_default_dtype(torch.bfloat16)
    with init_empty_weights():
        model = LlamaForCausalLM(config=model_config)
    model = accelerate.load_checkpoint_and_dispatch(model, model_args.model_name_or_path, device_map="auto", dtype=torch.bfloat16)
    model.bfloat16()
    test_loader_iter = iter(test_loader)
    results = []
    path = "./results/tora_13B_planning"
    with tqdm(total=164) as data:
        for ids, batch in zip(range(164), test_loader_iter):
            if os.path.exists(f"raw_tora_13B/HumanEval_results{ids}.json"):
                print(f"该问题已经解决{ids}，跳过")
                data.update(1)
                continue
            for k, v in batch.items():
                batch[k] = v.to(dist.get_rank())
            output = model.generate(inputs=batch['input_ids'],
                                    top_k=50,
                                    top_p=0.9,
                                    temperature=0.2,
                                    do_sample=True,
                                    max_new_tokens=512)
            data.update(1)
            with open(f"{path}/HumanEval_results{ids}.json", "w") as f:
                json.dump(dict(task_id=ids, completion=tokenizer.decode(output[0])), f)

def merge_json():
    import re
    from human_eval.data import write_jsonl, read_problems
    path = "raw_tora_13B"
    results = []
    problems = read_problems()
    task_ids = [task_id for task_id in problems]
    for task_name, ids in zip(task_ids, range(len(task_ids))):
        file = f"HumanEval_results{ids}.json"
        f = os.path.join(path, file)
        with open(f, "r") as fp:
            text = json.load(fp)
            text["task_id"] = task_name
            replace_text = re.findall(r"<s>\s*([^<\s].*?)\s*</s>|<s>[\n\r]*?.*?</s>", text["completion"],
                                      flags=re.DOTALL)
            if replace_text:
                text["completion"] = replace_text[0]
            else:
                replace_text = re.findall(r"(?<=<s>).*", text["completion"],
                                          flags=re.DOTALL)
                print(f"{ids}:{replace_text}")
                text["completion"] = replace_text[0]

            results.append(text)
    with open("samples.jsonl", "w") as f:
        for x in results:
            f.write(json.dumps(x) + "\n")



if __name__ == '__main__':
    inference("HumanEval")
    # merge_json()