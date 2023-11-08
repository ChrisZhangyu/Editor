import os
from typing import Iterator

from torch.utils.data import Dataset
import re


class CorrectDataset(Dataset):
    def __int__(self, path, data_type, mode="train"):
        if data_type == "APPS":
            pass
        elif data_type == "API":
            files = os.listdir(path)
            files = [os.path.join(path, item) for item in files]
            self.data = preprocess_for_api_data(files)
        else:
            raise Exception("不支持的数据集")


    def __getitem__(self, index):
        pass



    def __len__(self):
        pass


def preprocess_for_api_data(files):
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
            api_description = re.search(r"(?<=业务功能逻辑：).*?(?=需要检查的代码：)", text, flags=re.DOTALL)
            source_code = re.search(r"(?<=需要检查的代码：).*?(?=代码执行的错误信息：)", text, flags=re.DOTALL)
            source_code = source_code.group(0)
            java_code = re.findall(r"(?<=```java).*?(?=```)", source_code, flags=re.DOTALL)
            xml_code = re.findall(r"(?<=```xml).*?(?=```)", source_code, flags=re.DOTALL)

            break


if __name__ == '__main__':
    pass