from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import re
from transformers import BertTokenizer

# Load the IMDB movie review dataset
dataset=load_dataset('csv', data_files='IMDB Dataset.csv')
df = pd.DataFrame(dataset['train'])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

## Show the first few rows of the dataframe
#print(df.head())
#
## Check for missing values
#print("\nMissing values:\n", df.isnull().sum())
#
## Basic statistics of the dataset
#print("\nDataset statistics:\n", df.describe())
#
## Distribution of labels (class distribution)
#label_counts = df['sentiment'].value_counts()
#print("\nClass distribution:\n", label_counts)
#
## Plot the class distribution
#plt.figure(figsize=(6, 4))
#label_counts.plot(kind='bar', color=['skyblue', 'salmon'])
#plt.title('Class Distribution of IMDB Sentiment Dataset')
#plt.xlabel('Sentiment')
#plt.ylabel('Number of Samples')
#plt.show()

def clean_text(text):
    # Remove special characters, URLs, and excessive whitespaces
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return text.lower()

# Clean the reviews
df['cleaned_reviews'] = df['review'].apply(clean_text)

# Display the first few cleaned reviews
#print(df[['review', 'cleaned_reviews']].head())



# Tokenize and encode sequences in the dataset
encoded_inputs = tokenizer(
    df['cleaned_reviews'].tolist(),
    padding=True,  # Pad to the longest sequence in the batch
    truncation=True,  # Truncate long sentences to `max_length`
    max_length=128,  # Set a reasonable max length for BERT
    return_tensors='pt',  # Return PyTorch tensors
    add_special_tokens=True,  # Include [CLS] and [SEP] tokens
)

# Show example input IDs and attention masks
#print("Sample Encoded Inputs:", encoded_inputs['input_ids'][0])
#print("Sample Attention Masks:", encoded_inputs['attention_mask'][0])

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Convert labels to integers (BERT expects numerical labels)
label_dict = {'positive': 1, 'negative': 0}
df['label'] = df['sentiment'].map(label_dict)

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['cleaned_reviews'], df['label'], test_size=0.2, random_state=42)

# Tokenize the text data
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=128, return_tensors='pt')
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=128, return_tensors='pt')

# Convert labels to PyTorch tensors
train_labels = torch.tensor(train_labels.tolist())
val_labels = torch.tensor(val_labels.tolist())

# Create PyTorch Datasets
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)

# Create DataLoaders for training and validation
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=8)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=8)

print(f"Training and Validation DataLoaders created successfully!")

