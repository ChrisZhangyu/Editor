from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    log_interval = 5
    eval_interval = 200
    save_interval = 800
    batch_size = 1
    num_training_steps = 1000000
    weight_decay = 1e-1
    lr = 2e-4
    num_warmup_steps = 2000
    max_length = 4096


@dataclass
class ModelArgs:
    # model_name_or_path: Optional[str] = field(default=r"/root/autodl-tmp/Phind-CodeLlama-34B-v2")
    model_name_or_path: Optional[str] = field(default=r"/root/autodl-tmp/tora-code-13b")
    data_path: Optional[str] = field(default="/root/autodl-tmp/parallel/dataset/APPS/train")
