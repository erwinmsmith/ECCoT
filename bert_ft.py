import torch
from transformers import BertModel, BertTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR

class SentenceBert(torch.nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', output_dim=100):
        super(SentenceBert, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, output_dim)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 取 [CLS] token 的输出作为句子的嵌入
        sentence_embedding = outputs.last_hidden_state[:, 0, :]
        # 通过线性层调整维度
        sentence_embedding = self.linear(sentence_embedding)
        return sentence_embedding

def contrastive_loss(anchor, other, margin=1.0):
    dist = torch.norm(anchor - other, p=2, dim=-1)
    loss = torch.relu(dist - margin).mean()
    return loss

class TripleDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        a_text, b_text, c_text, label = self.data.iloc[idx]
        inputs_a = self.tokenizer(a_text, padding=False, truncation=True, max_length=self.max_length, return_token_type_ids=False)
        inputs_b = self.tokenizer(b_text, padding=False, truncation=True, max_length=self.max_length, return_token_type_ids=False)
        inputs_c = self.tokenizer(c_text, padding=False, truncation=True, max_length=self.max_length, return_token_type_ids=False)
        label = torch.tensor(label, dtype=torch.float32)
        return inputs_a, inputs_b, inputs_c, label

def custom_collate_fn(batch):
    inputs_a_list, inputs_b_list, inputs_c_list, labels = zip(*batch)
    
    # Pad all sequences in the batch to the same length
    max_len = max(len(item['input_ids']) for item in inputs_a_list + inputs_b_list + inputs_c_list)
    
    padded_inputs_a = []
    padded_attention_masks_a = []
    padded_inputs_b = []
    padded_attention_masks_b = []
    padded_inputs_c = []
    padded_attention_masks_c = []
    
    for inputs in inputs_a_list:
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        padded_input_ids = input_ids + [0] * (max_len - len(input_ids))
        padded_attention_mask = attention_mask + [0] * (max_len - len(attention_mask))
        padded_inputs_a.append(padded_input_ids)
        padded_attention_masks_a.append(padded_attention_mask)
    
    for inputs in inputs_b_list:
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        padded_input_ids = input_ids + [0] * (max_len - len(input_ids))
        padded_attention_mask = attention_mask + [0] * (max_len - len(attention_mask))
        padded_inputs_b.append(padded_input_ids)
        padded_attention_masks_b.append(padded_attention_mask)
    
    for inputs in inputs_c_list:
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        padded_input_ids = input_ids + [0] * (max_len - len(input_ids))
        padded_attention_mask = attention_mask + [0] * (max_len - len(attention_mask))
        padded_inputs_c.append(padded_input_ids)
        padded_attention_masks_c.append(padded_attention_mask)
    
    inputs_a = {
        'input_ids': torch.tensor(padded_inputs_a, dtype=torch.long),
        'attention_mask': torch.tensor(padded_attention_masks_a, dtype=torch.long)
    }
    inputs_b = {
        'input_ids': torch.tensor(padded_inputs_b, dtype=torch.long),
        'attention_mask': torch.tensor(padded_attention_masks_b, dtype=torch.long)
    }
    inputs_c = {
        'input_ids': torch.tensor(padded_inputs_c, dtype=torch.long),
        'attention_mask': torch.tensor(padded_attention_masks_c, dtype=torch.long)
    }
    labels = torch.tensor(labels, dtype=torch.float32)
    
    return inputs_a, inputs_b, inputs_c, labels

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df[['Q', 'R', 'A', 'label']]  # 仅读取前10行

def prepare_datasets_and_loaders(train_file, val_file, test_file, tokenizer, batch_size=16):
    train_data = load_data(train_file)
    val_data = load_data(val_file)
    test_data = load_data(test_file)

    train_dataset = TripleDataset(train_data, tokenizer)
    val_dataset = TripleDataset(val_data, tokenizer)
    test_dataset = TripleDataset(test_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    return train_loader, val_loader, test_loader

# 检查可用的 GPU 数量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载预训练的 BERT 和 Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 定义输出维度
output_dim = 100  # 你可以根据需要调整这个值

model = SentenceBert(output_dim=output_dim)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# 文件路径
train_file = 'output_train_results.csv'
val_file = 'output_val_results.csv'
test_file = 'output_test_results.csv'

# 设置 batch_size
batch_size = 64  # 你可以根据需要调整这个值

# 准备数据加载器
train_loader, val_loader, test_loader = prepare_datasets_and_loaders(train_file, val_file, test_file, tokenizer, batch_size=batch_size)

# 定义训练轮数
num_epochs = 500

# 设置早停参数
patience = 3
best_val_loss = float('inf')
early_stopping_counter = 0

# 学习率调度器
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()
    epoch_loss = 0.0
    
    for batch_idx, (inputs_a, inputs_b, inputs_c, labels) in enumerate(train_loader):
        inputs_a = {key: value.to(device) for key, value in inputs_a.items()}
        inputs_b = {key: value.to(device) for key, value in inputs_b.items()}
        inputs_c = {key: value.to(device) for key, value in inputs_c.items()}
        labels = labels.to(device)
        
        embedding_a = model(**inputs_a)
        embedding_b = model(**inputs_b)
        embedding_c = model(**inputs_c)
        
        ab_loss = contrastive_loss(embedding_a, embedding_b)
        bc_loss = contrastive_loss(embedding_b, embedding_c)
        
        # 使用 torch.where 处理批量标签
        loss = torch.where(labels == 1, ab_loss + bc_loss, torch.max(ab_loss, bc_loss)).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 累加损失
        epoch_loss += loss.item()
    
    # 计算平均损失
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Average Loss: {avg_epoch_loss:.4f}")

    # 验证阶段
    model.eval()
    val_epoch_loss = 0.0
    with torch.no_grad():
        for inputs_a, inputs_b, inputs_c, labels in val_loader:
            inputs_a = {key: value.to(device) for key, value in inputs_a.items()}
            inputs_b = {key: value.to(device) for key, value in inputs_b.items()}
            inputs_c = {key: value.to(device) for key, value in inputs_c.items()}
            labels = labels.to(device)
            
            embedding_a = model(**inputs_a)
            embedding_b = model(**inputs_b)
            embedding_c = model(**inputs_c)
            
            ab_loss = contrastive_loss(embedding_a, embedding_b)
            bc_loss = contrastive_loss(embedding_b, embedding_c)
            
            # 使用 torch.where 处理批量标签
            loss = torch.where(labels == 1, ab_loss + bc_loss, torch.max(ab_loss, bc_loss)).mean()
            
            # 累加损失
            val_epoch_loss += loss.item()
    
    # 计算平均损失
    avg_val_loss = val_epoch_loss / len(val_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Average Loss: {avg_val_loss:.4f}")

    # 检查是否需要早停
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stopping_counter = 0
        
        torch.save(model.state_dict(), 'sentence_bert_best_model.pth')
        print(f"Best validation loss updated to {best_val_loss:.4f}. Model saved.")
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Early stopping triggered.")
            break

# 测试阶段
print("Testing the model on the test set...")
model.load_state_dict(torch.load('sentence_bert_best_model.pth'))
model.eval()
test_epoch_loss = 0.0
with torch.no_grad():
    for inputs_a, inputs_b, inputs_c, labels in test_loader:
        inputs_a = {key: value.to(device) for key, value in inputs_a.items()}
        inputs_b = {key: value.to(device) for key, value in inputs_b.items()}
        inputs_c = {key: value.to(device) for key, value in inputs_c.items()}
        labels = labels.to(device)
        
        embedding_a = model(**inputs_a)
        embedding_b = model(**inputs_b)
        embedding_c = model(**inputs_c)
        
        ab_loss = contrastive_loss(embedding_a, embedding_b)
        bc_loss = contrastive_loss(embedding_b, embedding_c)
        
        # 使用 torch.where 处理批量标签
        loss = torch.where(labels == 1, ab_loss + bc_loss, torch.max(ab_loss, bc_loss)).mean()
        
        # 累加损失
        test_epoch_loss += loss.item()

# 计算平均损失
avg_test_loss = test_epoch_loss / len(test_loader)
print(f"Test Average Loss: {avg_test_loss:.4f}")

print("Training and testing complete!")



