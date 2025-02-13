import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import jieba
import chardet
import time
import torch
from torch.utils.data import Dataset, DataLoader
import gensim.models.keyedvectors as keyedvectors
from gensim.models import KeyedVectors
import bz2
import pickle

from utils import MRFDataset, calc_weight, split_dataset, clean_text, read_txt_files,tokenize,read_json_file
from model import MRF_ETM
from trainer_mrfetm import Trainer_MRF_ETM
from train_mrfetm import Train_MRF_ETM
from inference import inference
from edge2 import generate_edge_links

# 加载中文停用词表
with open('enstop.txt', 'r', encoding='utf-8') as f:
    chinese_stopwords = set(f.read().strip().split('\n'))
chinese_stopwords_list = list(chinese_stopwords) 

# 读取训练集、验证集和测试集文件夹中的所有txt文件
train_folder = 'ANLI/R1new/train.json'
val_folder = 'ANLI/R1new/dev.json'
test_folder = 'ANLI/R1new/test.json'

start_time = time.time()
print("Reading and processing text files...")
train_df = read_json_file(train_folder)
val_df = read_json_file(val_folder)
test_df = read_json_file(test_folder)
end_time = time.time()
print(f"Finished reading and processing text files. Time taken: {end_time - start_time:.2f} seconds")

# 文本清理
start_time = time.time()
print("Cleaning text...")
train_df['cleaned_text_d'] = train_df['text'].apply(clean_text)
val_df['cleaned_text_d'] = val_df['text'].apply(clean_text)
test_df['cleaned_text_d'] = test_df['text'].apply(clean_text)
end_time = time.time()
print(f"Finished cleaning text. Time taken: {end_time - start_time:.2f} seconds")

# 分词处理
start_time = time.time()
print("Tokenizing text...")
train_df['cleaned_text'] = train_df['cleaned_text_d'].apply(tokenize)
val_df['cleaned_text'] = val_df['cleaned_text_d'].apply(tokenize)
test_df['cleaned_text'] = test_df['cleaned_text_d'].apply(tokenize)
end_time = time.time()
print(f"Finished tokenizing text. Time taken: {end_time - start_time:.2f} seconds")

# 合并所有数据集以构建统一的词汇表
start_time = time.time()
print("Combining datasets to build vocabulary...")
combined_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
combined_df['words'] = combined_df['cleaned_text_d'].apply(tokenize)
end_time = time.time()
print(f"Finished combining datasets. Time taken: {end_time - start_time:.2f} seconds")

# 构建词汇表
start_time = time.time()
print("Building vocabulary...")
all_words = set(word for sublist in combined_df['words'] for word in sublist if word not in chinese_stopwords)
vocabulary = {word: idx for idx, word in enumerate(all_words)}
print("Vocabulary size:", len(vocabulary))
end_time = time.time()
print(f"Finished building vocabulary. Time taken: {end_time - start_time:.2f} seconds")

# 特征提取
vectorizer = CountVectorizer(stop_words=chinese_stopwords_list, vocabulary=vocabulary)

# 将分词后的词语列表重新组合成字符串
start_time = time.time()
print("Reconstructing text strings from tokens...")
train_df['cleaned_text_str'] = train_df['cleaned_text'].apply(lambda x: ' '.join(x))
val_df['cleaned_text_str'] = val_df['cleaned_text'].apply(lambda x: ' '.join(x))
test_df['cleaned_text_str'] = test_df['cleaned_text'].apply(lambda x: ' '.join(x))
end_time = time.time()
print(f"Finished reconstructing text strings. Time taken: {end_time - start_time:.2f} seconds")

# 使用相同的词汇表进行特征提取
start_time = time.time()
print("Extracting features using CountVectorizer...")
X_train = vectorizer.fit_transform(train_df['cleaned_text_str']).toarray()
X_val = vectorizer.transform(val_df['cleaned_text_str']).toarray()
X_test = vectorizer.transform(test_df['cleaned_text_str']).toarray()
end_time = time.time()
print(f"Finished extracting features. Time taken: {end_time - start_time:.2f} seconds")

# 获取特征名称
feature_names = vectorizer.get_feature_names_out()

# 打印特征矩阵的形状以确认一致性
print("Train feature matrix shape:", X_train.shape)
print("Validation feature matrix shape:", X_val.shape)
print("Test feature matrix shape:", X_test.shape)

# 创建DataFrame存储特征
start_time = time.time()
print("Creating DataFrames for features...")
train_features_df = pd.DataFrame(X_train, columns=feature_names)
train_features_df['batch_index'] = 0
train_features_df['file_path'] = train_df['file_path']

val_features_df = pd.DataFrame(X_val, columns=feature_names)
val_features_df['batch_index'] = 0
val_features_df['file_path'] = val_df['file_path']

test_features_df = pd.DataFrame(X_test, columns=feature_names)
test_features_df['batch_index'] = 0
test_features_df['file_path'] = test_df['file_path']
end_time = time.time()
print(f"Finished creating DataFrames. Time taken: {end_time - start_time:.2f} seconds")

# 确认特征矩阵的列数是否一致
print("Number of features in train set:", X_train.shape[1])
print("Number of features in validation set:", X_val.shape[1])
print("Number of features in test set:", X_test.shape[1])

embeddings_file = 'numberbatch-en.bz2'

# 初始化一个空的KeyedVectors对象
kv = KeyedVectors(vector_size=0)

start_time = time.time()
print("Loading pre-trained embeddings...")

with bz2.open(embeddings_file, 'rt', encoding='utf-8') as f:
    first_line = f.readline().strip()
    vocab_size, dim = map(int, first_line.split())
    
    # 创建一个临时列表来存储有效的词和向量
    valid_entries = []
    
    for line_no, line in enumerate(f):
        parts = line.strip().split()
        word = parts[0]
        vector = np.array(parts[1:], dtype=np.float32)
        
        if kv.vector_size == 0:
            kv.vector_size = len(vector)
        
        if len(vector) != kv.vector_size:
            print(f"Skipping word '{word}' with incorrect dimension {len(vector)} at line {line_no + 2}.")
            continue
        
        valid_entries.append((word, vector))
    
    # 设置最终的词汇表大小
    kv.vectors = np.zeros((len(valid_entries), kv.vector_size), dtype=np.float32)
    
    for index, (word, vector) in enumerate(valid_entries):
        kv.add_vector(word, vector)
# 将加载的词向量赋值给word_vectors
word_vectors = kv

end_time = time.time()
print(f"Finished loading embeddings. Time taken: {end_time - start_time:.2f} seconds")
print(f"Loaded {len(kv)} embeddings.")

# 生成边链接
edge_links = generate_edge_links(word_vectors, combined_df, vocabulary, edges_threshold=0.5)

# 保存边链接到文件
edges_file = 'edge_links.pkl'
with open(edges_file, 'wb') as f:
    pickle.dump(edge_links, f)

print(f"Edge links saved to {edges_file}")