import pandas as pd

def split_text(text):
    try:
        # 去除外层的双引号
        text = text.strip('"')
        
        # 查找 known_fact 开始和结束位置
        known_fact_start = text.find('"known_fact": "') + len('"known_fact": "')
        known_fact_end = text.find('", "hypothesis"', known_fact_start)
        known_fact = text[known_fact_start:known_fact_end].strip()
        
        # 查找 hypothesis 开始和结束位置
        hypothesis_start = text.find('"hypothesis": "') + len('"hypothesis": "')
        hypothesis_end = text.find('", "conclusion"', hypothesis_start)
        hypothesis = text[hypothesis_start:hypothesis_end].strip()
        
        # 查找 conclusion 开始和结束位置
        conclusion_start = text.find('"conclusion": "') + len('"conclusion": "')
        conclusion_end = text.find('"', conclusion_start)
        conclusion = text[conclusion_start:conclusion_end].strip()
        
        # 创建模板化的 'Q' 列
        q = f"Known {known_fact}. Based on the known information, determine if the assumption '{hypothesis}' is correct."
        
        return q, conclusion
    except Exception as e:
        print(f"Error processing text: {text}")
        print(e)
        return None, None

def process_csv(file_path, inplace=True):
    df = pd.read_csv(file_path)
    
    # 保留原始的 'text' 列，并添加新的列 'Q', 'A'
    df[['Q', 'A']] = df['text'].apply(lambda x: pd.Series(split_text(x)))
    
    # 决定是否覆盖原文件或另存为新文件
    output_file = file_path if inplace else f"processed_{file_path}"
    
    # 保存回 CSV 文件
    df.to_csv(output_file, index=False)
    print(f"Processed {file_path} and saved to {output_file}")

if __name__ == "__main__":
    csv_files = [
        'CALC_result/train_results_with_embeddings.csv',
        'CALC_result/val_results_with_embeddings.csv',
        'CALC_result/test_results_with_embeddings.csv'
    ]
    
    for file in csv_files:
        process_csv(file, inplace=True)  # 设为 False 可以保存到新文件



