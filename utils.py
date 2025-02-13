import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch import optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import re
import jieba

# Auxiliary functions
def calc_weight(
        epoch: int,
        n_epochs: int,
        cutoff_ratio: float = 0.,
        warmup_ratio: float = 1 / 3,
        min_weight: float = 0.,
        max_weight: float = 1e-5
) -> float:
    """Calculates weights.
    Args:
        epoch: current epoch.
        n_epochs: the total number of epochs to train the model.
        cutoff_ratio: ratio of cutoff epochs (set weight to zero) and
            n_epochs.
        warmup_ratio: ratio of warmup epochs and n_epochs.
        min_weight: minimum weight.
        max_weight: maximum weight.
    Returns:
        The current weight of the KL term.
    """

    fully_warmup_epoch = n_epochs * warmup_ratio

    if epoch < n_epochs * cutoff_ratio:
        return 0.
    if warmup_ratio:
        return max(min(1., epoch / fully_warmup_epoch) * max_weight, min_weight)
    else:
        return max_weight


# Custom Dataset class
class MRFDataset(Dataset):
    def __init__(self, features, batch_indices):
        self.features = features
        self.batch_indices = batch_indices

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        batch_index = self.batch_indices[idx]
        return feature, batch_index

# Function to split dataset into train and validation sets
def split_dataset(df, train_ratio=0.8):
    df = df.sample(frac=1).reset_index(drop=True)
    train_size = int(train_ratio * len(df))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    return train_df, val_df

# 文本清理函数
def clean_text(text):
    # 移除标点符号和数字
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # 转换为小写
    text = text.lower()
    return text
    
def tokenize(text):
    return list(jieba.cut(text))

def read_json_file(file_path):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # 使用正则表达式匹配每个条目
        entries = re.findall(r'\{([^}]+)\}', content)
        for index, entry in enumerate(entries):
            try:
                # 去除多余的逗号
                entry = re.sub(r',\s*$', '', entry)
                # 将整个entry转换为字符串
                text = entry.strip()
                # 去除空格和换行符
                cleaned_text = ' '.join(text.split())
                documents.append({'file_path': f"{file_path}:{index}", 'text': cleaned_text})
            except Exception as e:
                print(f"Error processing entry {index}: {e}")
    return pd.DataFrame(documents)


def read_txt_files(file_path):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()  # 去除首尾空格

    # 使用正则表达式匹配每个 { ... } 对象
    pattern = re.compile(r'\{[^{}]*\}')
    objects = pattern.findall(content)

    for obj in objects:
        # 去除对象内部的多余空格
        cleaned_obj = obj.strip()
        documents.append({'file_path': file_path, 'text': cleaned_obj})

    return pd.DataFrame(documents)

def read_txt_files2(file_path):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()  # 去除首尾空格

    # 使用正则表达式匹配每个 { ... } 对象
    pattern = re.compile(r'\{[^{}]*\}')
    objects = pattern.findall(content)

    for obj in objects:
        # 去除对象内部的多余空格
        cleaned_obj = obj.strip()

        # 提取问题的 stem、answer 和 choices
        stem_pattern = r'stem\s+is\s+([\s\S]*?)(?=,|answer)'
        answer_pattern = r'answer\s+is\s+([\s\S]*?)(?=,|\})'
        choices_pattern = r'choices\s+are\s+([\s\S]*?)(?= , the stem| the answer|\})'

        stem = re.search(stem_pattern, cleaned_obj).group(1).strip()
        answer = re.search(answer_pattern, cleaned_obj).group(1).strip()
        choices = re.search(choices_pattern, cleaned_obj).group(1).strip()

        # 根据 ', label' 分割 options，并去除多余的空格
        options = []
        for option in re.split(r',\s*label', choices):
            if option.strip():
                # 移除选项前后的多余空格
                option = option.strip()
                # 移除选项内容中的多余空格，如 'text :  bank' -> 'text :  bank' 变为 'text : bank'
                option = re.sub(r'\s*:\s*', ': ', option)
                options.append(option)

        # 组合成一句话
        reformatted_text = f"the stem is {stem}, the choices are {', label'.join(options)}, and the answer is {answer}"

        # 去除重组句子中的多余空格
        reformatted_text = re.sub(r'\s+', ' ', reformatted_text).strip()

        documents.append({'file_path': file_path, 'text': reformatted_text})

    return pd.DataFrame(documents)