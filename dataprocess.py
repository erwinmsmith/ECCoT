import json

# 标签映射
label_mapping = {
    "e": "entailment",
    "n": "neutral",
    "c": "contradiction"
}

# 读取JSON Lines文件
def read_json_lines_file(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

# 处理单个数据集条目
def process_data_entry(data):
    return {
        "known_fact": data["context"],
        "hypothesis": data["hypothesis"],
        "conclusion": label_mapping[data["label"]]
    }

# 主函数
def main():
    # 定义输入和输出文件路径
    input_files = ['ANLI/train.jsonl', 'ANLI/dev.jsonl', 'ANLI/test.jsonl']
    output_files = ['formatted_train.json', 'formatted_dev.json', 'formatted_test.json']

    for input_file, output_file in zip(input_files, output_files):
        # 读取JSON Lines文件
        data_sets = read_json_lines_file(input_file)

        # 转换数据格式
        formatted_data = [process_data_entry(data) for data in data_sets]

        # 将结果保存为JSON文件
        with open(output_file, 'w') as f:
            json.dump(formatted_data, f, indent=4)

        print(f"数据已成功转换并保存到 {output_file} 文件中")

if __name__ == "__main__":
    main()



