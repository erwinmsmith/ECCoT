import csv
import json

def read_csv(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return list(reader)

def process_data(data_list):
    processed_data = []
    for row in data_list:
        q = row['Q']
        a = row['A']
        r = row['reason']
        
        #instruction = "Determine if the input question is correct, provide the reason, and state 'entailmen' , 'neutra' or 'contradiction'."
        #instruction = "Perform the calculations based on the conditions and answer the questions."
        instruction = "Please select the most correct option."
        input_text = q
        output_text = f"because {r}, so the answer is {a}"
        
        processed_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })
    return processed_data

def save_to_json(data, file_name):
    with open(file_name, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

def main():
    # Read data from CSV files
    train_data = read_csv('Common_result/train_results_with_embeddings.csv')
    val_data = read_csv('Common_result/val_results_with_embeddings.csv')
    test_data = read_csv('Common_result/test_results_with_embeddings.csv')
    
    # Process data
    #processed_train_val_data = process_data(train_data + val_data)
    processed_train_val_data = process_data(train_data )
    processed_test_data = process_data(test_data)
    
    # Save to separate JSON files
    save_to_json(processed_train_val_data, 'Common_result/train_val_output.json')
    save_to_json(processed_test_data, 'Common_result/test_output.json')

if __name__ == "__main__":
    main()



