"""
@ Author: chen-chun-yan
@ Introduction: create correct model entity class codellama-7b
@ Create: 2023-11-3
@ Modify:
"""
import json
import os
import sys
import accelerate
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate import init_empty_weights
from datasets import IterableDataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, LlamaTokenizer, LlamaForCausalLM

from dataset import *

class CorrectModelClass:
    def __init__(self):
        self.batch_size = 2
        self.tokenizer = None
        self.model = None

    def build(self, model_path):
        try:
            # model_path = "llama-code-7B"
            # self.setup(0, 2)
            model_config = AutoConfig.from_pretrained(model_path, device="cuda")
            self.tokenizer = LlamaTokenizer.from_pretrained(model_path, device="cuda")
            self.tokenizer.mask_token = "[MASK]"
            self.tokenizer.sep_token = "[SEP]"
            self.tokenizer.cls_token = "[CLS]"
            self.tokenizer.pad_token = "\s"
            self.accelerator = Accelerator()
            torch.set_default_dtype(torch.bfloat16)
            with init_empty_weights():
                self.model = LlamaForCausalLM(config=model_config)


            gpu_num = 2
            # 构建正确的GPU映射，将参数分配到不同的GPU上
            decode_layers = model_config.num_hidden_layers
            group_length = decode_layers // gpu_num
            correct_device_map = {}
            start = 0

            for gpu_index in range(gpu_num):
                end = start + group_length
                for layer_index in range(start, end):
                    correct_device_map[f'model.layers.{layer_index}'] = gpu_index
                start = end




            # llama7B device_map样例
            # test_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0,
            #  'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0,
            #  'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0,
            #  'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 1,
            #  'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1,
            #  'model.layers.21': 1, 'model.layers.22': 1, 'model.layers.23': 1, 'model.layers.24': 1,
            #  'model.layers.25': 1, 'model.layers.26': 1, 'model.layers.27': 1, 'model.layers.28': 1,
            #  'model.layers.29': 1, 'model.layers.30': 1, 'model.layers.31': 1, 'model.norm': 1, 'lm_head': 1,
            #  }
            correct_device_map['model.embed_tokens'] = 0
            correct_device_map['lm_head'] = 1
            correct_device_map['model.norm'] = 1
            self.model = accelerate.load_checkpoint_and_dispatch(self.model, model_path, device_map=correct_device_map,
                                                                 dtype=torch.bfloat16)

            return 0
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            filename = exc_traceback.tb_frame.f_code.co_filename
            lineno = exc_traceback.tb_lineno
            print(f"An error occurred in {filename} at line {lineno}: {exc_type.__name__}: {exc_value}")
            return -1

    def generate(self, input_data):
        output_data = []
        # try:
        dataset = IterableDataset.from_generator(data_iterator,
                                                 gen_kwargs={'data_or_path': input_data,
                                                             'tokenizer': self.tokenizer,
                                                             'mode': 'inference',
                                                             'max_length': 2048})
        test_loader = DataLoader(dataset,
                                 collate_fn=collate_gen(self.tokenizer, "infer"),
                                 batch_size=self.batch_size,
                                 num_workers=1)
        test_loader_iter = iter(test_loader)
        for batch in test_loader_iter:
            for k, v in batch.items():
                batch[k] = v.to(self.accelerator.device)
            output = self.model.generate(**batch)
            output_data.append(self.tokenizer.decode(output[0]))
        return 0, output_data

        # except Exception as e:
        #     exc_type, exc_value, exc_traceback = sys.exc_info()
        #     filename = exc_traceback.tb_frame.f_code.co_filename
        #     lineno = exc_traceback.tb_lineno
        #     print(f"An error occurred in {filename} at line {lineno}: {exc_type.__name__}: {exc_value}")
        #     return -1, output_data

    def setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)


if __name__ == '__main__':
    model_path = "/root/autodl-tmp/code-llama-7b"
    correct_model = CorrectModelClass()
    ret = correct_model.build(model_path=model_path)

    if ret != 0:
        print("模型初始化错误")
    else:
        input_data = ["请输出冒泡排序"]
        ret, output_data = correct_model.generate(input_data)
        print(output_data)