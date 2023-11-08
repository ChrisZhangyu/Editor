import os
import torch
from human_eval.data import read_problems
from typing import Optional


def data_iterator(data_or_path,  tokenizer,  mode, max_length, dataset_type=Optional[str], process_index=0, num_processes=1, ):
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
                from few_shots import example
                problems[task_id]['prompt'] = example + "\n" + problems[task_id]['prompt']
                input_ids = tokenizer(problems[task_id]['prompt'])
                yield input_ids
        elif dataset_type == "API":
            import re
            path = data_or_path
            files = os.listdir(path)
            for file in files:
                with open(file, "r", encoding="utf-8") as f:
                    api_data = f.read()
                    instruct = "你是一个专业的程序员，下面的业务逻辑代码有错误，请你找出并修改"
                    prompt = api_data
                    llama_template = f'''[INST] <<SYS>>\n{instruct}\n<</SYS>>\n\n{prompt}[/INST]'''
                    input_ids = tokenizer(llama_template)
                    yield input_ids
                    # api_description = re.search(r"(?<=业务功能逻辑：).*?(?=需要检查的代码：)", api_data, flags=re.DOTALL)
                    # source_code = re.search(r"(?<=需要检查的代码：).*?(?=代码执行的错误信息：)", api_data, flags=re.DOTALL)
                    # source_code = source_code.group(0)
                    # java_code = re.findall(r"(?<=```java).*?(?=```)", source_code, flags=re.DOTALL)
                    # xml_code = re.findall(r"(?<=```xml).*?(?=```)", source_code, flags=re.DOTALL)

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
        for item in batch:
            input_ids.append(item['input_ids'])
            attention_mask.append(item['attention_mask'])
        if mode == "train":
            inputs = {
                'input_ids': torch.tensor(input_ids, dtype=torch.int64),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.int64),
                'labels': torch.tensor(input_ids[1:], dtype=torch.int64)
            }
        else:
            inputs = {
                'input_ids': torch.tensor(input_ids, dtype=torch.int64),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.int64),
            }

        return inputs
    return pretrain_collate_fn