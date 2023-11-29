import json
import os
import time
from typing import Callable

import accelerate
import colossalai.utils
import datasets
from accelerate import infer_auto_device_map
from colossalai.cluster import DistCoordinator
from colossalai.inference.tensor_parallel import TPInferEngine
from colossalai.lazy import LazyInitContext
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate import init_empty_weights
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin, GeminiPlugin
from colossalai.nn.optimizer import HybridAdam, FusedAdam
from colossalai.shardformer import ShardConfig
from colossalai.utils import get_current_device
from colossalai.zero import ColoInitContext, GeminiAdamOptimizer
from datasets import IterableDataset
from torch.optim import *
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, \
    get_cosine_schedule_with_warmup, AutoConfig
from config import *
from dataset import *
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

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


def train(plugin_type: str, is_lora=True):
    model_args, training_config = ModelArgs(), TrainConfig()
    # colossalai.launch_from_torch({})
    setup(0, 1)
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path, device="cuda")
    tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path, device="cuda")
    tokenizer.mask_token = "[MASK]"
    tokenizer.sep_token = "[SEP]"
    tokenizer.cls_token = "[CLS]"
    tokenizer.pad_token = "\s"
    coordinator = DistCoordinator()
    dataset = IterableDataset.from_generator(data_iterator, gen_kwargs={'data_or_path': model_args.data_path,
                                                                        'tokenizer': tokenizer,
                                                                        'mode': 'train',

                                                                        'dataset_type': 'API'})


    train_loader = DataLoader(dataset,
                              collate_fn=collate_gen(tokenizer, training_config.max_length),
                              batch_size=training_config.batch_size,
                              num_workers=1)

    torch.set_default_dtype(torch.bfloat16)

    coordinator.print_on_master("加载模型")
    print(colossalai.utils.get_current_device())

    with LazyInitContext(default_device=get_current_device()):
        raw_model = LlamaForCausalLM(config=model_config)

    coordinator.print_on_master("准备阶段")

    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in raw_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_config.weight_decay,
        },
        {
            "params": [p for n, p in raw_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0,
        }
    ]


    coordinator.print_on_master("加载优化器")
    optimizer = HybridAdam(optimizer_grouped_parameters, lr=training_config.lr, betas=(0.9, 0.95))
    # optimizer = FusedAdam(optimizer_grouped_parameters, lr=training_config.lr, betas=(0.9, 0.95))
    # 为了使所有GPU的学习率与单GPU情况下一致
    # factor = accelerator.num_processes / accelerator.gradient_accumulation_steps
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=training_config.num_warmup_steps * factor,
    #                                             num_training_steps=training_config.num_training_steps * factor)

    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=training_config.num_warmup_steps,
                                                num_training_steps=training_config.num_training_steps)
    coordinator.print_on_master("并行初始化")

    if plugin_type == "HybridParallel":
        coordinator.print_on_master("HybridParallel")

        plugin = HybridParallelPlugin(tp_size=1,
                                      pp_size=1,
                                      microbatch_size=1,
                                      cpu_offload=True,
                                      enable_all_optimization=True,
                                      zero_stage=2,
                                      precision='bf16',
                                      initial_scale=1)
    elif plugin_type == "gemini":
        coordinator.print_on_master("gemini")
        plugin = GeminiPlugin(
            placement_policy="auto",
            precision="bf16",
            verbose=True,
            steady_cuda_cap_ratio=1.0,
            search_range_m=256,
            # offload_optim_frac=1.0,
            # offload_param_frac=1.0,
            growth_factor=1,
            hidden_dim=model_config.hidden_size,
        )
    else:
        raise RuntimeError("不支持的插件类型")

    writer = SummaryWriter("./log")
    booster = Booster(plugin=plugin, mixed_precision="")
    dist.get_rank()

    model, optimizer, criterion, train_dataloader, scheduler = booster.boost(raw_model, optimizer, nn.CrossEntropyLoss
                                                                             , train_loader, scheduler)
    booster.load_model(model, model_args.model_name_or_path)
    if is_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32,
                                 lora_dropout=0.1, target_modules=["q_proj", "k_proj", "v_proj"])
        model = get_peft_model(model.module, peft_config)
        model.print_trainable_parameters()
    torch.cuda.synchronize()
    for epoch in range(1):
        one_epoch(epoch, model, optimizer, criterion, scheduler, train_dataloader, booster, writer, coordinator)
        # save_checkpoint()
    writer.flush()





def one_epoch(epoch,
              model: nn.Module,
              optimizer: Optimizer,
              criterion: Callable,
              scheduler: LRScheduler,
              train_dataloader: DataLoader,
              booster: Booster,
              writer: SummaryWriter,
              coordinator: DistCoordinator):
    # print(type(train_dataloader))
    # total_step = len(train_dataloader)
    total_step = 206
    train_loader_iter = iter(train_dataloader)
    coordinator.print_on_master("开始训练")
    model.train()
    optimizer.zero_grad()
    loss = 206
    with tqdm(range(total_step),
              desc=f'Epoch [{epoch + 1}]') as pbar:
        for i in pbar:
            batch = next(train_loader_iter)
            for k, v in batch.items():
                batch[k] = v.to(torch.cuda.current_device())
            outputs = model(**batch)
            loss = outputs[0]
            writer.add_scalar("Loss/train", loss, i)
            writer.flush()
            # time.sleep(5)
            booster.backward(loss, optimizer)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


def save_checkpoint(epoch: int,
                    training_config: TrainConfig,
                    model: nn.Module,
                    optimizer: Optimizer,
                    scheduler: LRScheduler,
                    booster: Booster,
                    step: int,
                    batch_size: int,
                    coordinator: DistCoordinator
                    ):
    full_path = os.path.join(checkpoint_save_path, "checkpoints")
    if not os.path.isdir(full_path):
        os.mkdir(full_path)
    booster.save_model(model, os.path.join(full_path, "model"), shard=True)
    booster.save_optimizer(optimizer, os.path.join(full_path, "optimizer"), shard=True)
    booster.save_lr_scheduler(scheduler, os.path.join(full_path, "lr_scheduler"))
    running_states = {"epoch": epoch, "step": step, "sample_start_index": step * batch_size, }
    if coordinator.is_master():
       with open(os.path.join(full_path, "running_states.json"), "w") as f:
           json.dump(running_states, f)



def inference(dataset_type, dataloader):
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

    torch.set_default_dtype(torch.bfloat16)
    with init_empty_weights():
        model = LlamaForCausalLM(config=model_config)
    model = accelerate.load_checkpoint_and_dispatch(model, model_args.model_name_or_path, device_map="auto",
                                                    dtype=torch.bfloat16)

    dataset = IterableDataset.from_generator(data_iterator,
                                             gen_kwargs={'data_or_path': model_args.data_path, 'tokenizer': tokenizer,
                                                         'dataset_type': dataset_type, 'mode': 'train',
                                                         'max_length': 1024})


    test_loader = DataLoader(dataset, collate_fn=collate_gen(tokenizer, "infer"), batch_size=training_config.batch_size,
                             num_workers=1)
    model.bfloat16()
    # shard_config = ShardConfig(enable_tensor_parallelism=True, inference_only=True)
    # infer_engine = TPInferEngine(model, shard_config, training_config.batch_size, 2048,
    #                              2048)
    test_loader_iter = iter(test_loader)
    results = []
    with tqdm(total=164) as data:
        for ids, batch in zip(range(164), test_loader_iter):
            if os.path.exists(f"{inference_save_path}/HumanEval_results{ids}.json"):
                print(f"该问题已经解决{ids}，跳过")
                data.update(1)
                continue
            for k, v in batch.items():
                batch[k] = v.to(dist.get_rank())
            # output = infer_engine.generate(inputs=batch['input_ids'],
            #                       top_k=50,
            #                       top_p=0.9,
            #                       temperature=0.2,
            #                       do_sample=True,
            #                       max_new_tokens=512)
            # 生成计划
            plans = model.generate(inputs=batch['input_ids'],
                                    # top_p=0.9,
                                    # temperature=0.,
                                    # top_k=50,
                                    # do_sample=False,
                                    max_new_tokens=512)
            # 根据计划完成代码
            # print(tokenizer.decode(plans[0]))
            data.update(1)
            with open(f"{inference_save_path}/HumanEval_results{ids}.json", "w") as f:
                json.dump(dict(task_id=ids, completion=tokenizer.decode(plans[0])), f)



def merge_json():
    import re
    from human_eval.data import write_jsonl, read_problems
    from few_shots import example4_plan

    results = []
    problems = read_problems()
    task_ids = [task_id for task_id in problems]
    for task_name, ids in zip(task_ids, range(len(task_ids))):
        file = f"HumanEval_results{ids}.json"
        f = os.path.join(inference_save_path, file)
        with open(f, "r") as fp:
            text = json.load(fp)
            text["task_id"] = task_name

            if example4_plan in text['completion']:
                text['completion'] = text['completion'].replace(example4_plan, '')

            # 取<s>与</s>之间的字符串
            replace_text = re.findall(r"<s>\s*([^<\s].*?)\s*</s>|<s>[\n\r]*?.*?</s>", text["completion"],
                                      flags=re.DOTALL)


            # 部分代码没有</s>
            if replace_text:
                text["completion"] = replace_text[0]
            else:
                replace_text = re.findall(r"(?<=<s>).*", text["completion"],
                                          flags=re.DOTALL)
                print(f"{ids}:{replace_text}")
                text["completion"] = replace_text[0]
            # 如果生成了多个def则取第一个
            between_def = re.split(r'def',  text["completion"], flags=re.DOTALL)
            if between_def:
                text["completion"] = between_def[0] + "def" + between_def[1]

            print(f"****************************{task_ids}")
            print(text['completion'])
            results.append(text)
    with open("samples.jsonl", "w") as f:
        for x in results:
            f.write(json.dumps(x) + "\n")


if __name__ == '__main__':
    # inference("HumanEval", "")
    train("gemini")
    # merge_json()
