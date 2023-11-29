from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    log_interval: int = 5
    eval_interval: int = 200
    save_interval: int = 800
    batch_size: int = 1
    num_training_steps: int = 1000000
    weight_decay: int = 1e-1
    lr: int = 2e-4
    num_warmup_steps: int = 2000
    max_length: int = 4096


@dataclass
class ModelArgs:
    # model_name_or_path: Optional[str] = field(default=r"/root/autodl-tmp/Phind-CodeLlama-34B-v2")
    # model_name_or_path: Optional[str] = field(default=r"/root/autodl-tmp/tora-code-13b")
    model_name_or_path: Optional[str] = field(default=r"/root/autodl-tmp/code-llama-7b")
    # data_path: Optional[str] = field(default="/root/autodl-tmp/parallel/dataset/APPS/train")
    data_path: Optional[str] = field(default="/root/autodl-tmp/Editor/api_data")
