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

from utils import MRFDataset, calc_weight, split_dataset, clean_text, read_txt_files,tokenize,read_json_file,read_txt_files2
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
train_folder = 'Common/train.txt'
val_folder = 'Common/dev.txt'
test_folder = 'Common/dev.txt'

start_time = time.time()
print("Reading and processing text files...")
train_df = read_txt_files2(train_folder)
val_df = read_txt_files2(val_folder)
test_df = read_txt_files2(test_folder)
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

"""
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
#print(edge_links)
"""

edge_links=[(1,2),(11,17)]
# 创建datasets和dataloaders
train_dataset = MRFDataset(features=torch.tensor(train_features_df.drop(columns=['batch_index', 'file_path']).values, dtype=torch.float32),
                           batch_indices=torch.tensor(train_features_df['batch_index'].values, dtype=torch.long))
val_dataset = MRFDataset(features=torch.tensor(val_features_df.drop(columns=['batch_index', 'file_path']).values, dtype=torch.float32),
                         batch_indices=torch.tensor(val_features_df['batch_index'].values, dtype=torch.long))

train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# 测试集的dataset和dataloader
test_dataset = MRFDataset(features=torch.tensor(test_features_df.drop(columns=['batch_index', 'file_path']).values, dtype=torch.float32),
                          batch_indices=torch.tensor(test_features_df['batch_index'].values, dtype=torch.long))
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# 模型参数
input_dim = X_train.shape[1]
num_topic = 50
emd_dim = 100
num_batch = 256

# 初始化模型, 优化器和trainer
encoder, decoder, optimizer = MRF_ETM(input_dim, num_batch, num_topic, emd_dim)
trainer = Trainer_MRF_ETM(encoder, decoder, optimizer)

# 训练循环
Train_MRF_ETM(trainer, train_dataloader, val_dataloader, Total_epoch=1000, edge_links=None)

# 使用 inference 函数对训练集数据进行推理
train_data = torch.tensor(train_features_df.drop(columns=['batch_index', 'file_path']).values, dtype=torch.float32).to('cuda')
train_batch_indices = torch.tensor(train_features_df['batch_index'].values, dtype=torch.long).to('cuda')
train_inferred_theta, train_recon_log_mod, train_topic_embeddings, train_word_embeddings, train_topic_word_distribution = inference(trainer, train_data, train_batch_indices)

# 使用 inference 函数对验证集数据进行推理
val_data = torch.tensor(val_features_df.drop(columns=['batch_index', 'file_path']).values, dtype=torch.float32).to('cuda')
val_batch_indices = torch.tensor(val_features_df['batch_index'].values, dtype=torch.long).to('cuda')
val_inferred_theta, val_recon_log_mod, val_topic_embeddings, val_word_embeddings, val_topic_word_distribution = inference(trainer, val_data, val_batch_indices)

# 使用 inference 函数对测试集数据进行推理
test_data = torch.tensor(test_features_df.drop(columns=['batch_index', 'file_path']).values, dtype=torch.float32).to('cuda')
test_batch_indices = torch.tensor(test_features_df['batch_index'].values, dtype=torch.long).to('cuda')
test_inferred_theta, test_recon_log_mod, test_topic_embeddings, test_word_embeddings, test_topic_word_distribution = inference(trainer, test_data, test_batch_indices)

# 打印推理结果的形状
print("Inferred Train Doc-Topic Distribution:", train_inferred_theta.shape)
print("Inferred Train Topic-Word Distribution:", train_topic_word_distribution.shape)
print("Inferred Train Topic Embeddings:", train_topic_embeddings.shape)
print("Inferred Train Word Embeddings:", train_word_embeddings.shape)
print("Inferred Train Recon Log Mod:", train_recon_log_mod.shape)

print("Inferred Val Doc-Topic Distribution:", val_inferred_theta.shape)
print("Inferred Val Topic-Word Distribution:", val_topic_word_distribution.shape)
print("Inferred Val Topic Embeddings:", val_topic_embeddings.shape)
print("Inferred Val Word Embeddings:", val_word_embeddings.shape)
print("Inferred Val Recon Log Mod:", val_recon_log_mod.shape)

print("Inferred Test Doc-Topic Distribution:", test_inferred_theta.shape)
print("Inferred Test Topic-Word Distribution:", test_topic_word_distribution.shape)
print("Inferred Test Topic Embeddings:", test_topic_embeddings.shape)
print("Inferred Test Word Embeddings:", test_word_embeddings.shape)
print("Inferred Test Recon Log Mod:", test_recon_log_mod.shape)

# 获取每个文档的主题标签
train_doc_topic_labels = np.argmax(train_inferred_theta, axis=1)
train_features_df['topic_label'] = train_doc_topic_labels

val_doc_topic_labels = np.argmax(val_inferred_theta, axis=1)
val_features_df['topic_label'] = val_doc_topic_labels

test_doc_topic_labels = np.argmax(test_inferred_theta, axis=1)
test_features_df['topic_label'] = test_doc_topic_labels

# 获取每个主题的Top Words
top_words_per_topic = {}
for topic_idx in range(num_topic):
    top_words = [feature_names[i] for i in train_topic_word_distribution[topic_idx].argsort()[-10:][::-1]]
    top_words_per_topic[f'topic_{topic_idx}'] = ', '.join(top_words)

# 将结果保存到DataFrame
train_results_df = train_features_df.copy()
train_results_df['doc_topic_distribution'] = list(train_inferred_theta)
train_results_df['topic_top_words'] = train_results_df['topic_label'].map(lambda x: top_words_per_topic[f'topic_{x}'])

val_results_df = val_features_df.copy()
val_results_df['doc_topic_distribution'] = list(val_inferred_theta)
val_results_df['topic_top_words'] = val_results_df['topic_label'].map(lambda x: top_words_per_topic[f'topic_{x}'])

test_results_df = test_features_df.copy()
test_results_df['doc_topic_distribution'] = list(test_inferred_theta)
test_results_df['topic_top_words'] = test_results_df['topic_label'].map(lambda x: top_words_per_topic[f'topic_{x}'])

# 将原始样本（text列）添加到结果DataFrame中
train_results_df['text'] = train_df['text']
val_results_df['text'] = val_df['text']
test_results_df['text'] = test_df['text']

# 打印结果DataFrame的前几行，包括原始样本
print(train_results_df[['file_path', 'topic_label', 'topic_top_words', 'text']].head())
print(val_results_df[['file_path', 'topic_label', 'topic_top_words', 'text']].head())
print(test_results_df[['file_path', 'topic_label', 'topic_top_words', 'text']].head())

# 保存结果DataFrame到CSV文件
#train_results_df[['file_path', 'topic_label', 'topic_top_words', 'text']].to_csv('ANLI_result/train_results.csv', index=False)
#val_results_df[['file_path', 'topic_label', 'topic_top_words', 'text']].to_csv('ANLI_result/val_results.csv', index=False)
#test_results_df[['file_path', 'topic_label', 'topic_top_words', 'text']].to_csv('ANLI_result/test_results.csv', index=False)

# 1. 提取每个主题的嵌入向量
topic_embeddings = {
    f'topic_{i}': train_topic_embeddings[i].tolist() for i in range(num_topic)
}

# 2. 提取每个主题Top Words的嵌入向量
top_words_vectors_per_topic = {}
for topic_idx in range(num_topic):
    top_word_indices = train_topic_word_distribution[topic_idx].argsort()[-10:][::-1]
    top_words_vectors = train_word_embeddings[top_word_indices]
    top_words_vectors_per_topic[f'topic_{topic_idx}'] = top_words_vectors.tolist()

# 3. 将主题嵌入向量和Top Words嵌入向量添加到结果DataFrame中
def add_embeddings_to_df(df, topic_label_col, topic_embeddings, top_words_vectors_per_topic):
    # 添加主题嵌入向量
    df['topic_embedding'] = df[topic_label_col].apply(lambda x: topic_embeddings[f'topic_{int(x)}'])
    
    # 添加Top Words嵌入向量
    df['top_words_vectors'] = df[topic_label_col].apply(lambda x: top_words_vectors_per_topic[f'topic_{int(x)}'])
    return df

train_results_df = add_embeddings_to_df(train_results_df[['file_path', 'topic_label', 'topic_top_words', 'text']], 'topic_label', topic_embeddings, top_words_vectors_per_topic)
val_results_df = add_embeddings_to_df(val_results_df[['file_path', 'topic_label', 'topic_top_words', 'text']], 'topic_label', topic_embeddings, top_words_vectors_per_topic)
test_results_df = add_embeddings_to_df(test_results_df[['file_path', 'topic_label', 'topic_top_words', 'text']], 'topic_label', topic_embeddings, top_words_vectors_per_topic)

# 4. 保存更新后的结果DataFrame到CSV文件
train_results_df.to_csv('Common_result/train_results_with_embeddings.csv', index=False)
val_results_df.to_csv('Common_result/val_results_with_embeddings.csv', index=False)
test_results_df.to_csv('Common_result/test_results_with_embeddings.csv', index=False)