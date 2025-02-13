import pandas as pd
from ollama import Client  # Assuming you have installed the ollama library
import shutil
import os

# Clean up control characters in the API response
def clean_text(text):
    # You can add more cleanup logic here if needed
    return text.replace('\n', ' ').replace('\r', '').strip()

# Function to call the Ollama API and get a summary for a topic label based on multiple texts
def get_topic_summary(topic_label, texts, client, batch_size=1000):
    try:
        print(f"Generating summary for topic: {topic_label}")
        summaries = []
        total_texts = len(texts)
        
        # Split texts into batches
        for i in range(0, total_texts, batch_size):
            batch_texts = texts[i:i + batch_size]
            text_summary = " ".join(batch_texts)
            prompt = (
                f"Given the topic label '{topic_label}', provide a concise summary of what this topic means in natural language"
                f"Please provide only the general meaning of this topic without analyzing each text individually. "
                f"and only focus on giving a refined summary of the topic itself:\n"
                f"Texts: {text_summary}\n\n"
                f"The given label is '{topic_label}'. It is related to the theme of [insert relevant theme here]."
            )
            
            response = client.chat(
                model="llama3.3",  # Assuming llama3.3 is the correct model name for Ollama
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.6, "max_tokens": 1024}  # Adjust based on the expected length of the summary
            )
            
            # Clean up the API response
            cleaned_response = clean_text(response.message.content)
            summaries.append(cleaned_response)
        
        final_summary = " ".join(summaries)
        print(f"Summary generated for topic: {topic_label}")
        print(f"Summary: {final_summary}\n")  # Print the generated summary
        return final_summary
    except Exception as e:
        print(f"Failed to get summary for topic {topic_label}: {e}")
        return "Failed to retrieve summary"

# Initialize the Ollama API client
client = Client(host="http://127.0.0.1:11434")  # Replace with the actual host and port if different

# Define the input CSV files
csv_files = [
    'Common_result/train_results_with_embeddings.csv',
    'Common_result/val_results_with_embeddings.csv',
    'Common_result/test_results_with_embeddings.csv'
]

# Process each CSV file and update it with the summarized topic explanations
for csv_file in csv_files:
    print(f"Processing file: {csv_file}")
    
    # Read the CSV file
    df = pd.read_csv(csv_file, encoding='utf-8')
    print(f"File {csv_file} read successfully.")
    
    # Remove rows where Q or A is empty
    filtered_df = df.dropna(subset=['Q', 'A'])
    print(f"Filtered out rows with empty Q or A columns. Remaining rows: {len(filtered_df)}")
    
    # Create a dictionary to store the topic summaries
    topic_summaries = {}

    # Collect unique topic labels and their corresponding texts
    for topic_label, group in filtered_df.groupby('topic_label'):
        print(f"Processing topic: {topic_label}")
        if topic_label not in topic_summaries:
            texts = group['Q'].dropna().tolist() + group['A'].dropna().tolist()
            if texts:
                summary = get_topic_summary(topic_label, texts, client)
                topic_summaries[topic_label] = summary
            else:
                topic_summaries[topic_label] = "No texts available for this topic"
                print(f"No texts available for topic: {topic_label}")

    # Add the topic explanation as a new column
    filtered_df['topic_explanation'] = filtered_df['topic_label'].map(topic_summaries)
    print("Added topic explanations as a new column.")

    # Save the updated DataFrame back to the original CSV file
    filtered_df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"Updated data successfully written back to {csv_file}")

print("All datasets have been processed with topic summaries.")