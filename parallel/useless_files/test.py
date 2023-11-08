from parallel.useless_files.temp import convert_llama_hg2pp
import accelerate
import deepspeed.comm
from deepspeed.runtime.pipe import LayerSpec
from torch.utils.data import DataLoader
from config import *
from datasets import IterableDataset
from dataset import *
# import wandb
from transformers import AutoTokenizer, LlamaConfig, \
    get_cosine_schedule_with_warmup
from deepspeed.ops.adam import FusedAdam
from deepspeed.pipe import PipelineModule

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print("PyTorch 版本:", torch.__version__)


def config_deepspeed(accelerator):
    accelerator.state.num_processes = 2
    accelerator.state.downcast_bfloat = False
    accelerator.state.distributed_type = accelerate.DistributedType.DEEPSPEED
    deepspeed_config = {
        "zero_optimization": {
            "stage": 1,
            "offload_optimizer_device": "none",
            "offload_param_device": "none",

        },
        "train_batch_size": 16,
        "train_micro_batch_size_per_gpu": "auto",
        "bf16": {
            "enabled": True
        },
        "deepspeed_multinode_launcher": "standard",
        "gradient_accumulation_steps": 4,
        "gradient_clipping": 1.0,
    }
    hf_deepspeed_config = accelerate.utils.deepspeed.HfDeepSpeedConfig(deepspeed_config)
    deepspeed_plugin = accelerate.DeepSpeedPlugin(hf_ds_config=hf_deepspeed_config)
    accelerator.state.deepspeed_plugin = deepspeed_plugin


def train():

    model_args, training_config = ModelArgs(), TrainConfig()

    model_config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.mask_token = "[MASK]"
    tokenizer.sep_token = "[SEP]"
    tokenizer.cls_token = "[CLS]"
    dataset = IterableDataset.from_generator(data_iterator, gen_kwargs={'data_path': model_args.data_path,
                                                                        'tokenizer': tokenizer})
    train_loader = DataLoader(dataset, collate_fn=collate_gen(tokenizer), batch_size=training_config.batch_size,
                              num_workers=1)
    # 有些属性没设置
    # 存疑，需要试一下
    print("加载模型")

   


    deepspeed_config = {
        "zero_optimization": {
            "stage": 1,
            "offload_optimizer_device": "none",
            "offload_param_device": "none",

        },
        "train_batch_size": 16,
        "train_micro_batch_size_per_gpu": "auto",
        "bf16": {
            "enabled": True
        },

        "deepspeed_multinode_launcher": "standard",
        "gradient_accumulation_steps": 4,
        "gradient_clipping": 1.0,
    }
    resutl = convert_llama_hg2pp(raw_model.state_dict())

    deepspeed.init_distributed()
    spec_layers = [LayerSpec(layer) for layer in raw_model.model.layers]
    raw_model = PipelineModule(layers=spec_layers, num_stages=2)
    print(raw_model.hf_device_map)

    raw_model.eval()
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
            "params": [p for n, p in raw_model.named_parameters() if any(nd for nd in no_decay)],
            "weight_decay": 0,
        }
    ]
    optimizer = FusedAdam(optimizer_grouped_parameters, lr=training_config.lr, betas=(0.9, 0.95))
    optimizer.zero_grad()
    # 为了使所有GPU的学习率与单GPU情况下一致
    # factor = accelerator.num_processes / accelerator.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=training_config.num_warmup_steps,
                                                num_training_steps=training_config.num_training_steps)

    model, optimizer, _, scheduler = deepspeed.initialize(model=raw_model,
                                                          optimizer=optimizer,
                                                          lr_scheduler=scheduler,
                                                          config=deepspeed_config)

    train_loader_iter = iter(train_loader)
    print("开始训练")
    global_step = 0
    for data_step in range(training_config.num_training_steps):
        print(f"第{data_step}")
        model.train()

        # 使用迭代器方式得到本批次训练数据
        batch = next(train_loader_iter)
        # batch是字典类型，包括了input_id，和attention mask

        labels = batch['input_ids'].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        # 输入input_ids和attention mask
        # with torch.no_grad():
        #     output = model(**batch, labels=labels)
        #     print(output)
        output = model(**batch, labels=labels)
        total_loss = output.loss
        optimizer.backward(total_loss)
        # 在梯度累积的步骤内更新参数，hf上给的例子就是这样的
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()





    print("debug")


if __name__ == '__main__':
    train()
