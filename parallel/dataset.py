import json
import os
from functools import partial

import torch
from transformers import LlamaTokenizer

from Editor.parallel.config import ModelArgs, TrainConfig
from human_eval.data import read_problems
from typing import Optional


def data_iterator(data_or_path,  tokenizer,  mode, dataset_type=Optional[str], max_length=2048,  process_index=0, num_processes=1, ):
    assert data_or_path
    # 首先确定训练或者推理
    if mode == "train":
        if dataset_type == "APPS":
            path = data_or_path
            files = os.listdir(path)
            pad_id = tokenizer.pad_token_id
            for i, file in enumerate(files):
                if num_processes > 1 and i % process_index != process_index:
                    continue
                complete_path = os.path.join(os.path.join(path, file), "question.txt")
                with open(complete_path, 'r') as f:
                    _input = f.read()
                    input_ids = tokenizer(_input)
                    pad_length = max_length - len(input_ids['input_ids'])
                    input_ids['input_ids'] = [pad_id] * pad_length + input_ids['input_ids']
                    input_ids['attention_mask'] = [0] * pad_length + input_ids['attention_mask']
                    yield input_ids
        elif dataset_type == "HumanEval":
            problems = read_problems()
            for task_id in problems:
                from few_shots import example4_plan
                # print(problems[task_id]['prompt'])
                # print("*"*100)
                question = problems[task_id]['prompt'][:-4]

                problems[task_id]['prompt'] = example4_plan + "\n" + question + 'Let’s think step by step\n"""'
                input_ids = tokenizer(problems[task_id]['prompt'])
                yield input_ids
        elif dataset_type == "API":
            print("API")
            path = data_or_path
            prompt_path = os.path.join(path, 'correct_prompt')
            truth_path = os.path.join(path, 'ground_truth')
            prompts = sorted(os.listdir(prompt_path))
            truths = sorted(os.listdir(truth_path))
            for prompt, truth in zip(prompts, truths):
                prompt_full_path = os.path.join(prompt_path, prompt)
                truth_full_path = os.path.join(truth_path, truth)
                try:
                    with open(prompt_full_path, "r", encoding="utf-8") as fp, open(truth_full_path, "r", encoding="utf-8") as ft:
                        prompt_str = fp.read()
                        truth_str = ft.read()
                except UnicodeDecodeError:
                        print(f"{prompt_full_path}或{truth_full_path}文件编码错误，读取失败")
                instruct = "你是一个专业的程序员，下面的业务逻辑代码有错误，请你找出并修改"
                llama_template = f'''[INST] <<SYS>>\n{instruct}\n<</SYS>>\n\n{prompt_str}\n\n{truth_str}[/INST]'''

                input_ids = tokenizer(llama_template)
                yield input_ids
        else:
            raise Exception("不支持的数据集")
    else:
        data = data_or_path
        for item in data:
            input_ids = tokenizer(item)
            yield input_ids


def collate_gen(tokenizer, segment_max_length, mode="train", ):
    # dataloader 在从dataset中将数据组织成batch后传入这个函数
    def pretrain_collate_fn(batch):
        input_ids = []
        attention_mask = []
        # 手动组成batch
        for item in batch:
            input_ids.append(item['input_ids'])
            attention_mask.append(item['attention_mask'])
        if mode == "train":
            inputs = {
                'input_ids': torch.tensor(input_ids, dtype=torch.int64),
                # 'attention_mask': torch.tensor(attention_mask, dtype=torch.int64),
                # transformers低层会自动进行label偏移，因此这里只需要复制input
                'labels': torch.tensor(input_ids, dtype=torch.int64)
            }
        else:
            inputs = {
                'input_ids': torch.tensor(input_ids, dtype=torch.int64),
                # 'attention_mask': torch.tensor(attention_mask, dtype=torch.int64),
            }

        return inputs
    return pretrain_collate_fn


def spilt_prompt(api_data):
    import re
    api_description = re.search(r"(?<=业务功能逻辑：).*?(?=需要检查的代码：)", api_data, flags=re.DOTALL)
    source_code = re.search(r"(?<=需要检查的代码：).*?(?=代码执行的错误信息：)", api_data, flags=re.DOTALL)
    source_code = source_code.group(0)
    java_code = re.findall(r"(?<=```java).*?(?=```)", source_code, flags=re.DOTALL)
    xml_code = re.findall(r"(?<=```xml).*?(?=```)", source_code, flags=re.DOTALL)

if __name__ == '__main__':
    model_args, training_config = ModelArgs(), TrainConfig()
    tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path, device="cuda")
    data_iter = data_iterator(model_args.data_path, tokenizer, 'train', 'API')
    count = 0
    sum = 0
    max = 0
    min = 100000000
    json_prompt = []
    for i,  in data_iter:
        count += 1
        length = len(i['input_ids'])
        print(f"{count}：{length}")
        sum += length
        max = length if length > max else max
        min = length if length < min else min
        json_prompt.append({f'{count}': length})
    json_prompt.append({'max': max,
                        'min': min,
                        'average': sum/count})

    with open("./prompt_length_statistics.jsonl", "w") as f:
        for item in json_prompt:
            f.write(json.dumps(item) + "\n")
    print(f"平均长度为：{sum/count}, 最大长度为：{max}, 最小长度为：{min}")