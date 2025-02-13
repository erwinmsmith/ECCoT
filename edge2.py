import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch import optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
import gensim.models.keyedvectors as keyedvectors
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import chardet

def get_cosine(vec1, vec2):
    """计算两个向量之间的余弦相似度"""
    return cosine_similarity([vec1], [vec2])[0][0]
    
def generate_edge_links(word_vectors, df, vocabulary, edges_threshold=0.8):
    """
    生成边链接列表
    
    :param word_vectors: 预训练的词向量对象
    :param df: 包含文档单词列表的 DataFrame
    :param vocabulary: 单词到索引的映射字典
    :param edges_threshold: 余弦相似度阈值，默认为 0.8
    :return: 边链接列表
    """
    # 初始化边字典
    edge_dict = defaultdict(list)

    # 获取预训练词嵌入中的所有词
    pretrained_words = set(word_vectors.index_to_key)

    # 更新词汇表以仅包含预训练词嵌入中存在的词
    updated_vocabulary = {word: idx for idx, word in enumerate(vocabulary) if word in pretrained_words}

    # 反转词汇表以便通过索引查找词
    reverse_vocabulary = {idx: word for word, idx in updated_vocabulary.items()}

    # 确保文档中的词都在更新后的词汇表中
    filtered_df = df.copy()
    filtered_df['cleaned_text'] = filtered_df['cleaned_text'].apply(lambda doc: [word for word in doc if word in updated_vocabulary])

    # 遍历每个文档，生成边链接
    for idx, doc in enumerate(filtered_df['cleaned_text'].values):
        doc_set = set(doc)  # 使用集合去重
        doc_vectors = {word: word_vectors[word] for word in doc_set}
        
        # 检查是否有有效的词向量
        if not doc_vectors:
            print(f"No valid vectors found for document {idx}. Skipping.")
            continue
        
        # 计算余弦相似度矩阵
        words = list(doc_vectors.keys())
        vectors = np.array([doc_vectors[word] for word in words])
        
        # 检查 vectors 是否为空
        if len(vectors) == 0:
            print(f"Vectors array is empty for document {idx}. Skipping.")
            continue
        
        similarity_matrix = cosine_similarity(vectors)
        
        for i, word_i in enumerate(words):
            for j, word_j in enumerate(words):
                if i == j:
                    continue
                sim = similarity_matrix[i, j]
                if sim > edges_threshold:
                    edge_dict[updated_vocabulary[word_i]].append(updated_vocabulary[word_j])

    # 将边字典转换为边链接列表
    edge_links = [(k, v) for k, values in edge_dict.items() for v in values]
    print('success changed')
    return edge_links