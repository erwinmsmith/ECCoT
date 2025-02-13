import pandas as pd
from ollama import Client  # Assuming you have installed the ollama library
import shutil
import os

# Clean up control characters in the API response
def clean_text(text):
    return text.replace('\n', ' ').replace('\r', '').strip()

# Function to call the local ollama model
def call_local_ollama(prompt, client, model_name="llama3.3"):
    try:
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.6, "max_tokens": 1024}  # Adjust parameters as needed
        )
        
        # Extract the content from the message
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            cleaned_response = clean_text(response.message.content)
        else:
            cleaned_response = "Unexpected API response format, please check the logs."
        
        return cleaned_response
    except Exception as e:
        print(f"Local API call failed: {e}")
        return "API call failed, please check the logs."

# Initialize the local ollama client
client = Client(host="http://127.0.0.1:11434")  # Replace with the actual host and port if different

# Define the input CSV files (assuming the file paths are correct)
csv_files = [
    'Common_result/train_results_with_embeddings.csv',
    'Common_result/val_results_with_embeddings.csv',
    'Common_result/test_results_with_embeddings.csv'
]

# Process each CSV file and update it with the new columns
for csv_file in csv_files:
    # Create a backup of the original file
    backup_file = csv_file + '.bak'
    shutil.copy(csv_file, backup_file)
    print(f"Backup created: {backup_file}")

    # Read the CSV file
    df = pd.read_csv(csv_file, encoding='utf-8')
    
    # Remove rows where Q or Topic explanation is empty
    filtered_df = df.dropna(subset=['Q', 'topic_explanation'])

    for index, row in filtered_df.iterrows():
        instruction = row['Q']
        topic_explanation = row['topic_explanation']
        
        prompt = (
            f"Given the following information:\n"
            f"  - Description: {instruction}\n"
            f"  - Topic explanation: {topic_explanation}\n"
            f"\n"
            f"Let's analyze the question '{instruction}' step by step in light of the given topic explanation\n"
            f"Ensure that your answer is accurate, high-quality, logical, and concise."
        )
        
        # Call the local ollama model to get the reason
        reason = call_local_ollama(prompt, client)

        print(f"Question: {instruction}")
        print(f"Reason: {reason}\n")
        
        # Add the reason and label as new columns
        filtered_df.at[index, 'reason'] = reason
        filtered_df.at[index, 'label'] = 1  # Always set label to 1
    
    # Save the updated DataFrame back to the original CSV file
    filtered_df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"Data successfully written to {csv_file}")

print("All datasets have been processed.")



