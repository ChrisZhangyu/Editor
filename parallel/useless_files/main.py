import os
import time
from dataclasses import dataclass
import accelerate
import deepspeed.comm
from deepspeed import PipelineModule
from deepspeed.ops.adam import FusedAdam
from torch.utils.data import DataLoader
from torch.optim import *
from torchinfo import summary
from config import *
import torch
from accelerate import Accelerator, infer_auto_device_map, notebook_launcher
from datasets import IterableDataset
from dataset import *
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
# import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, \
    HfArgumentParser, get_cosine_schedule_with_warmup, AutoConfig
from deepspeed.ops.adam import FusedAdam
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print("PyTorch 版本:", torch.__version__)

def config_deepspeed(accelerator):
    accelerator.state.num_processes = 4
    accelerator.state.downcast_bfloat = False
    accelerator.state.distributed_type = accelerate.DistributedType.DEEPSPEED
    deepspeed_config = {
        "zero_optimization": {
            "stage": 1,
            "offload_optimizer_device": "none",
            "offload_param_device": "none",

        },
        "train_micro_batch_size_per_gpu": "auto",
        "bf16": {
            "enabled": True
        },
        "deepspeed_multinode_launcher": "standard",
        "gradient_accumulation_steps": 12,
        "gradient_clipping": 1.0,

    }


    hf_deepspeed_config = accelerate.utils.deepspeed.HfDeepSpeedConfig(deepspeed_config)
    deepspeed_plugin = accelerate.DeepSpeedPlugin(hf_ds_config=hf_deepspeed_config)
    accelerator.state.deepspeed_plugin = deepspeed_plugin

def train():
    accelerator = Accelerator()
    config_deepspeed(accelerator)

    parser = HfArgumentParser((ModelArgs, TrainConfig,))
    model_args, training_config = parser.parse_args_into_dataclasses()


    training_config.log_interval *= accelerator.gradient_accumulation_steps
    training_config.eval_interval *= accelerator.gradient_accumulation_steps
    training_config.save_interval *= accelerator.gradient_accumulation_steps

    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.mask_token = "[MASK]"
    tokenizer.sep_token = "[SEP]"
    tokenizer.cls_token = "[CLS]"
    dataset = IterableDataset.from_generator(data_iterator, gen_kwargs={'data_path': model_args.data_path,
                                                                        'tokenizer': tokenizer})
    train_loader = DataLoader(dataset,  collate_fn=collate_gen(tokenizer), batch_size=training_config.batch_size, num_workers=1)
    # 有些属性没设置
    # 存疑，需要试一下
    print("加载模型")

    with init_empty_weights():
        raw_model = AutoModelForCausalLM.from_config(model_config)

    # device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0,
    #               'model.layers.3': 0,
    #               'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0,
    #               'model.layers.8': 0,
    #               'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 1, 'model.layers.12': 1,
    #               'model.layers.13': 1,
    #               'model.layers.14': 1, 'model.layers.15': 1, 'model.layers.16': 1, 'model.layers.17': 1,
    #               'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1,
    #               'model.layers.22': 1,
    #               'model.layers.23': 2, 'model.layers.24': 2, 'model.layers.25': 2, 'model.layers.26': 2,
    #               'model.layers.27': 2, 'model.layers.28': 2, 'model.layers.29': 2, 'model.layers.30': 2,
    #               'model.layers.31': 2,
    #               'model.layers.32': 2, 'model.layers.33': 2, 'model.layers.34': 2, 'model.layers.35': 2,
    #               'model.layers.37': 3,
    #               'model.layers.38': 3, 'model.layers.39': 3, 'model.layers.40': 3, 'model.layers.41': 3,
    #               'model.layers.42': 3,
    #               'model.layers.43': 3, 'model.layers.44': 3, 'model.layers.45': 3, 'model.layers.46': 3,
    #               'model.layers.47': 3,
    #               'model.norm': 3, 'lm_head': 3, 'model.layers.36': 3}
    device_map = infer_auto_device_map(raw_model)
    new_device_map = {}
    for k, v in device_map.items():
        import re
        num = re.search(r'\d+', k)

        num = int(num.group()) if num else -1
        if num == 17:
            new_device_map["model.layers.17"] = 0
        else:
            if 23 <= num :
                new_device_map[k] = 1
            elif num < 23:
                new_device_map[k] = 0
            # elif num >= 32:
            #     new_device_map[k] = 2

            else:
                new_device_map[k] = v

    raw_model = load_checkpoint_and_dispatch(raw_model, model_args.model_name_or_path,
                                             device_map=new_device_map,
                                             dtype=torch.bfloat16)
    print(raw_model.hf_device_map)


    # raw_model.eval()
    # 会导致oom
    # with torch.no_grad():
    #     summary(raw_model.cuda(), input_data=torch.ones(1, 1, dtype=torch.int16).cuda())
    # code_llama中没找到bias
    print("准备阶段")

    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in raw_model.named_parameters() if not any(nd for nd in no_decay)],
            "weight_decay": training_config.weight_decay,
        },
        {
            "params": [p for n, p in raw_model.named_parameters() if  any(nd for nd in no_decay)],
            "weight_decay": 0,
        }
    ]
    optimizer = SGD(optimizer_grouped_parameters, lr=training_config.lr)
    optimizer.zero_grad()
    # 为了使所有GPU的学习率与单GPU情况下一致
    factor = accelerator.num_processes / accelerator.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=training_config.num_warmup_steps * factor,
                                                num_training_steps=training_config.num_training_steps * factor)

    train_loader, model, optimizer, scheduler = accelerator.prepare(train_loader,
                                                                       raw_model, optimizer, scheduler)
    train_loader_iter = iter(train_loader)
    print("开始训练")
    global_step = 0
    for data_step in range(training_config.num_training_steps):
        print(f"第{data_step}")
        model.train()
        with accelerator.accumulate(model):
            # 使用迭代器方式得到本批次训练数据
            batch = next(train_loader_iter)
            # batch是字典类型，包括了input_id，和attention mask
            for k, v in batch.items():
                batch[k] = v.to(accelerator.device)
            labels = batch['input_ids'].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            # 输入input_ids和attention mask
            with torch.no_grad():
                output = model(**batch, labels=labels)
                print(output)
            output = model(**batch, labels=labels)
            total_loss = output.loss
            accelerator.backward(total_loss)
            # 在梯度累积的步骤内更新参数，hf上给的例子就是这样的
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if accelerator.sync_gradients:
                global_step += 1
        if data_step % training_config.log_interval == 0 and data_step > 0 and accelerator.is_main_process:
            cost_time = time.time() - start_time
            start_time = time.time()
            tokens = training_config.train_batch_size * training_config.log_interval * 1500
            # wandb.log({'Training/Token per second per gpu': tokens / cost_time})

            current_lr = optimizer.param_groups[0]['lr']

            accelerator.print('Global Step: {}, Data Step: {}, Loss: {}, Token per second per gpu: {}'.format(
                global_step, data_step, total_loss, tokens / cost_time))

    print("debug")

if __name__ == '__main__':

    train()