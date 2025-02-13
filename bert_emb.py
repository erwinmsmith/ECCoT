import torch
from transformers import BertModel, BertTokenizer
import pandas as pd
from torch.utils.data import DataLoader

class SentenceBert(torch.nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', output_dim=100):
        super(SentenceBert, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, output_dim)
        self.output_dim = output_dim  # 添加 output_dim 属性
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 取 [CLS] token 的输出作为句子的嵌入
        sentence_embedding = outputs.last_hidden_state[:, 0, :]
        # 通过线性层调整维度
        sentence_embedding = self.linear(sentence_embedding)
        return sentence_embedding

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def generate_embeddings(model, tokenizer, file_paths, base_columns, batch_size=4):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        df = load_data(file_path)
        
        # 获取所有列名
        all_columns = list(df.columns)
        
        # 确定 base_columns 在列中的索引
        try:
            q_index = all_columns.index(base_columns[0])
            a_index = all_columns.index(base_columns[1])
            topic_explanation_index = all_columns.index(base_columns[2])
        except ValueError as e:
            print(f"Error: {e}. Please ensure that the specified columns exist in the DataFrame.")
            continue
        
        # 获取 base_columns 之后的所有列
        additional_columns = all_columns[topic_explanation_index + 1:]
        
        # 过滤掉为空的列
        non_empty_columns = []
        for col in additional_columns:
            if not df[col].isnull().all():
                non_empty_columns.append(col)
        
        # 组合所有需要嵌入的列
        columns_to_embed = base_columns + non_empty_columns
        
        # 创建一个字典来存储每一列的嵌入
        all_embeddings = {col: [] for col in columns_to_embed}
        
        for index, row in df.iterrows():
            for col in columns_to_embed:
                text = str(row[col]) if pd.notna(row[col]) else ''
                
                if text:
                    inputs = tokenizer(text, padding=False, truncation=True, max_length=128, return_token_type_ids=False)
                    
                    dataset = [inputs]
                    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: custom_collate_fn(x))
                    
                    with torch.no_grad():
                        for batch in dataloader:
                            batch = {k: v.to(device) for k, v in batch.items()}
                            embedding = model(**batch).cpu().numpy()[0]
                            all_embeddings[col].append(embedding)
                else:
                    all_embeddings[col].append([0.0] * model.output_dim)  # 使用零向量填充缺失值
        
        # Combine embeddings into a single DataFrame
        combined_embeddings = {}
        for col in columns_to_embed:
            combined_embeddings[f'emb_{col}'] = all_embeddings[col]
        
        embeddings_df = pd.DataFrame(combined_embeddings)
        
        # Merge original DataFrame with embeddings DataFrame
        merged_df = pd.concat([df.reset_index(drop=True), embeddings_df], axis=1)
        
        # Save the merged DataFrame back to the original file
        merged_df.to_csv(file_path, index=False)

def custom_collate_fn(batch):
    max_len = max(len(item['input_ids']) for item in batch)
    
    padded_input_ids = []
    padded_attention_masks = []
    
    for item in batch:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        padded_input_ids.append(input_ids + [0] * (max_len - len(input_ids)))
        padded_attention_masks.append(attention_mask + [0] * (max_len - len(attention_mask)))
    
    return {
        'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(padded_attention_masks, dtype=torch.long)
    }

# 加载预训练的 BERT 和 Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 定义输出维度
output_dim = 100  # 你可以根据需要调整这个值

# 初始化 SentenceBert 模型并加载最佳权重
model = SentenceBert(output_dim=output_dim)
model.load_state_dict(torch.load('sentence_bert_best_model.pth'))

# 文件路径
file_paths = ['Common_result/train_results_with_embeddings.csv', 'Common_result/val_results_with_embeddings.csv', 'Common_result/test_results_with_embeddings.csv']

# 基础列名列表
base_columns = ['Q', 'A', 'topic_explanation']

# 设置 batch_size
batch_size = 64  # 你可以根据需要调整这个值

# 生成嵌入并保存到原始 CSV 文件
generate_embeddings(model, tokenizer, file_paths, base_columns, batch_size=batch_size)



