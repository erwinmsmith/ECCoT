import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 定义函数来读取CSV文件并提取特定的嵌入特征列
def load_and_extract_features(file_path, feature_prefix):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 提取所有匹配feature_prefix的列名
    feature_columns = [col for col in df.columns if col.startswith(feature_prefix)]
    
    # 初始化一个空列表来存储所有的特征向量
    features_list = []
    
    # 遍历每一行数据
    for index, row in df.iterrows():
        # 对于每一列特征，解析其内容并添加到features_list中
        for col in feature_columns:
            vector_str = row[col]
            # 去除多余的空格和换行符，并分割成单独的数值字符串
            vector_values = vector_str.replace('[', '').replace(']', '').split()
            # 进一步清理每个数值字符串，去除逗号
            vector_values = [value.strip(',') for value in vector_values]
            # 将数值字符串转换为浮点数
            try:
                vector = [float(value) for value in vector_values]
                features_list.append(vector)
            except ValueError as e:
                print(f"Error converting to float: {e}")
                print(f"Offending string: {vector_str}")
                raise
    
    # 将features_list转换为NumPy数组
    features_array = np.array(features_list).reshape(-1, len(feature_columns), 100)
    
    return features_array

# 定义函数来读取CSV文件并提取特定的嵌入特征列（适用于单个向量）
def load_and_extract_single_feature(file_path, column_name):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 提取指定列的数据
    vectors = df[column_name].apply(lambda x: np.array(eval(x))).tolist()
    
    # 将列表转换为NumPy数组
    vectors_array = np.array(vectors)
    
    return vectors_array

# 文件路径
train_file_path = 'Common_result/train_results_with_embeddings.csv'
val_file_path = 'Common_result/val_results_with_embeddings.csv'
test_file_path = 'Common_result/test_results_with_embeddings.csv'

# 特征前缀
feature_prefix_emb_reason_sentence = 'emb_reason_sentence_'
feature_prefix_emb_Q = 'emb_Q'
feature_prefix_emb_A = 'emb_A'

# 加载并提取特征
train_features_emb_reason_sentence = load_and_extract_features(train_file_path, feature_prefix_emb_reason_sentence)
val_features_emb_reason_sentence = load_and_extract_features(val_file_path, feature_prefix_emb_reason_sentence)
test_features_emb_reason_sentence = load_and_extract_features(test_file_path, feature_prefix_emb_reason_sentence)

train_features_emb_Q = load_and_extract_features(train_file_path, feature_prefix_emb_Q)
val_features_emb_Q = load_and_extract_features(val_file_path, feature_prefix_emb_Q)
test_features_emb_Q = load_and_extract_features(test_file_path, feature_prefix_emb_Q)

train_features_emb_A = load_and_extract_features(train_file_path, feature_prefix_emb_A)
val_features_emb_A = load_and_extract_features(val_file_path, feature_prefix_emb_A)
test_features_emb_A = load_and_extract_features(test_file_path, feature_prefix_emb_A)

# 提取 topic_embedding 列并转换为 NumPy 数组
train_topic_embeddings = load_and_extract_single_feature(train_file_path, 'topic_embedding')
val_topic_embeddings = load_and_extract_single_feature(val_file_path, 'topic_embedding')
test_topic_embeddings = load_and_extract_single_feature(test_file_path, 'topic_embedding')

# 提取 top_words_vectors 列并转换为 NumPy 数组
train_top_words_vectors = load_and_extract_single_feature(train_file_path, 'top_words_vectors')
val_top_words_vectors = load_and_extract_single_feature(val_file_path, 'top_words_vectors')
test_top_words_vectors = load_and_extract_single_feature(test_file_path, 'top_words_vectors')

# 检查维度
print("Train emb_reason_sentence features shape:", train_features_emb_reason_sentence.shape)
print("Validation emb_reason_sentence features shape:", val_features_emb_reason_sentence.shape)
print("Test emb_reason_sentence features shape:", test_features_emb_reason_sentence.shape)

print("Train emb_Q features shape:", train_features_emb_Q.shape)
print("Validation emb_Q features shape:", val_features_emb_Q.shape)
print("Test emb_Q features shape:", test_features_emb_Q.shape)

print("Train emb_A features shape:", train_features_emb_A.shape)
print("Validation emb_A features shape:", val_features_emb_A.shape)
print("Test emb_A features shape:", test_features_emb_A.shape)

print("Train topic_embeddings 的维度:", train_topic_embeddings.shape)
print("Validation topic_embeddings 的维度:", val_topic_embeddings.shape)
print("Test topic_embeddings 的维度:", test_topic_embeddings.shape)

print("Train top_words_vectors 的维度:", train_top_words_vectors.shape)
print("Validation top_words_vectors 的维度:", val_top_words_vectors.shape)
print("Test top_words_vectors 的维度:", test_top_words_vectors.shape)

# 定义函数来计算余弦相似度并生成数据表
def calculate_cosine_similarities(emb_reason_sentence, emb_Q, emb_A, topic_embeddings, top_words_vectors):
    num_samples = emb_reason_sentence.shape[0]
    num_sentences = emb_reason_sentence.shape[1]
    
    # 初始化结果列表
    results = []
    
    for i in range(num_samples):
        sample_results = {}
        
        # 计算 emb_reason_sentence 与其他向量的余弦相似度
        for j in range(num_sentences):
            sentence = emb_reason_sentence[i, j]
            
            # 检查句子是否为零向量
            if np.allclose(sentence, np.zeros_like(sentence)):
                sample_results[f'emb_reason_sentence_{j}'] = {
                    'cosine_sim_with_emb_Q': np.nan,
                    'cosine_sim_with_emb_A': np.nan,
                    'cosine_sim_with_topic_embedding': np.nan,
                    'cosine_sim_with_top_words_vectors': np.nan,
                    'average_cosine_similarity': np.nan
                }
                continue
            
            sentence = sentence.reshape(1, -1)
            
            # 检查 emb_Q 是否为零向量
            if np.allclose(emb_Q[i], np.zeros_like(emb_Q[i])):
                sim_Q = np.nan
            else:
                sim_Q = cosine_similarity(sentence, emb_Q[i].reshape(1, -1)).mean()
            
            # 检查 emb_A 是否为零向量
            if np.allclose(emb_A[i], np.zeros_like(emb_A[i])):
                sim_A = np.nan
            else:
                sim_A = cosine_similarity(sentence, emb_A[i].reshape(1, -1)).mean()
            
            # 检查 topic_embedding 是否为零向量
            if np.allclose(topic_embeddings[i], np.zeros_like(topic_embeddings[i])):
                sim_topic = np.nan
            else:
                sim_topic = cosine_similarity(sentence, topic_embeddings[i].reshape(1, -1)).mean()
            
            # 检查 top_words_vectors 是否为零向量
            if np.allclose(top_words_vectors[i], np.zeros_like(top_words_vectors[i])):
                sim_top_words = np.nan
            else:
                sim_top_words = cosine_similarity(sentence, top_words_vectors[i]).mean()
            
            # 计算平均相似度
            valid_sims = [sim for sim in [sim_Q, sim_A, sim_topic, sim_top_words] if not np.isnan(sim)]
            if valid_sims:
                avg_sim = np.mean(valid_sims)
            else:
                avg_sim = np.nan
            
            sample_results[f'emb_reason_sentence_{j}'] = {
                'cosine_sim_with_emb_Q': sim_Q,
                'cosine_sim_with_emb_A': sim_A,
                'cosine_sim_with_topic_embedding': sim_topic,
                'cosine_sim_with_top_words_vectors': sim_top_words,
                'average_cosine_similarity': avg_sim
            }
        
        # 展平字典并将结果添加到总结果列表中
        for key, value in sample_results.items():
            results.append({
                'sample_index': i,
                'sentence_index': int(key.split('_')[-1]),
                **value
            })
    
    # 创建 DataFrame
    df = pd.DataFrame(results)
    
    return df

# 计算训练集的余弦相似度
train_df = calculate_cosine_similarities(
    train_features_emb_reason_sentence,
    train_features_emb_Q.squeeze(),
    train_features_emb_A.squeeze(),
    train_topic_embeddings,
    train_top_words_vectors
)

# 计算验证集的余弦相似度
val_df = calculate_cosine_similarities(
    val_features_emb_reason_sentence,
    val_features_emb_Q.squeeze(),
    val_features_emb_A.squeeze(),
    val_topic_embeddings,
    val_top_words_vectors
)

# 计算测试集的余弦相似度
test_df = calculate_cosine_similarities(
    test_features_emb_reason_sentence,
    test_features_emb_Q.squeeze(),
    test_features_emb_A.squeeze(),
    test_topic_embeddings,
    test_top_words_vectors
)

# 打印部分结果以验证
print("Train Data Frame Sample:")
print(train_df.head())

print("\nValidation Data Frame Sample:")
print(val_df.head())

print("\nTest Data Frame Sample:")
print(test_df.head())

# 去掉包含任何空值的行
train_df_cleaned = train_df.dropna(how='any')
val_df_cleaned = val_df.dropna(how='any')
test_df_cleaned = test_df.dropna(how='any')

# 将数据集保存为CSV文件
train_df_cleaned.to_csv('common_train_data_cos.csv', index=False)
val_df_cleaned.to_csv('common_validation_data_cos.csv', index=False)
test_df_cleaned.to_csv('common_test_data_cos.csv', index=False)