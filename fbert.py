import pandas as pd
import numpy as np

# Define the input CSV files and their corresponding output CSV file names
csv_files = {
    'Common_result/train_results_with_embeddings.csv': 'output_train_results.csv',
    'Common_result/val_results_with_embeddings.csv': 'output_val_results.csv',
    'Common_result/test_results_with_embeddings.csv': 'output_test_results.csv'
}

# Process each CSV file and create new datasets
for csv_file, output_file in csv_files.items():
    # Read the CSV file
    df = pd.read_csv(csv_file, encoding='utf-8')
    
    # Ensure the required columns exist
    if not {'Q', 'A', 'label'}.issubset(df.columns):
        print(f"Required columns ('Q', 'A', 'label') do not exist in {csv_file}. Skipping this file.")
        continue
    
    # Extract all columns after 'label'
    label_index = df.columns.get_loc('label')
    feature_columns = df.columns[label_index + 1:]
    
    # Create a list to store the new data
    new_data = []
    
    for index, row in df.iterrows():
        q = row['Q']
        a = row['A']
        label = row['label']
        
        for feature_col in feature_columns:
            feature_value = row[feature_col]
            if pd.notna(feature_value) and len(str(feature_value).strip()) > 2:  # Check if the feature value is not NaN and length > 2
                new_row = {
                    'Q': q,
                    'A': a,
                    'label': label,
                    'R': feature_value
                }
                new_data.append(new_row)
    
    # Create a DataFrame from the new data
    new_df = pd.DataFrame(new_data)
    
    # Randomly shuffle the DataFrame
    new_df = new_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Get the number of rows in the original DataFrame
    num_rows_original = len(new_df)
    
    # Generate additional samples with label 0
    additional_samples = []
    while len(additional_samples) < num_rows_original:
        sample_indices = np.random.choice(num_rows_original, size=1, replace=False)[0]
        sampled_row = new_df.iloc[sample_indices]
        
        q = sampled_row['Q']
        a = sampled_row['A']
        r = sampled_row['R']
        
        new_sample = {
            'Q': q,
            'A': a,
            'label': 0,
            'R': r
        }
        additional_samples.append(new_sample)
    
    # Create a DataFrame from the additional samples
    additional_df = pd.DataFrame(additional_samples)
    
    # Concatenate the original and additional DataFrames
    final_df = pd.concat([new_df, additional_df], ignore_index=True)
    
    # Shuffle the final DataFrame
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the final DataFrame to the output CSV file
    final_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Data successfully written to {output_file}")

print("All datasets have been processed.")



