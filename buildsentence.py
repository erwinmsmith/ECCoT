import pandas as pd
from pathlib import Path

# Define the input CSV files
csv_files = {
    'Common_result/train_results_with_embeddings.csv': 'output_train_results.csv',
    'Common_result/val_results_with_embeddings.csv': 'output_val_results.csv',
    'Common_result/test_results_with_embeddings.csv': 'output_test_results.csv'
}

# Process each CSV file and update it with the new columns
for csv_file in csv_files.keys():
    # Read the CSV file
    df = pd.read_csv(csv_file, encoding='utf-8')
    
    # Ensure the 'reason' column exists
    if 'reason' not in df.columns:
        print(f"Column 'reason' does not exist in {csv_file}. Skipping this file.")
        continue
    
    # Split the 'reason' column into sentences based on '.'
    df['reason'] = df['reason'].astype(str).str.replace('\n', ' ').str.replace('\r', '')
    sentences = df['reason'].str.split(r'\.\s*', expand=True)
    
    # Add the sentences as new columns
    for i in range(sentences.shape[1]):
        col_name = f'reason_sentence_{i+1}'
        df[col_name] = sentences[i]
    
    # Save the updated DataFrame back to the original CSV file
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"Data successfully written to {csv_file}")

print("All datasets have been processed.")



