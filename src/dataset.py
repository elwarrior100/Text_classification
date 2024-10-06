from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import re
from transformers import BertTokenizer
import torch

# Load the IMDB movie review dataset
dataset=load_dataset('csv', data_files='IMDB Dataset.csv')
df = pd.DataFrame(dataset['train'])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

from transformers import BertForSequenceClassification

# Load pre-trained BERT with a classification head
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',  # Pre-trained BERT model
    num_labels=2  # Number of classes (positive and negative sentiment)
)

# Print the model architecture
print(model)



import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the optimizer (AdamW is recommended for BERT)
optimizer = AdamW(model.parameters(), lr=2e-5)  # Learning rate is usually small for BERT fine-tuning

# Loss function (CrossEntropyLoss is used for classification tasks)
loss_fn = torch.nn.CrossEntropyLoss()

# Create the DataLoaders (use the ones created in Step 2)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

print("DataLoaders and optimizer initialized successfully!")


# Training function
def train(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs=2):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Training phase
        model.train()  # Set model to training mode
        total_loss = 0

        for batch in train_dataloader:
            # Unpack the inputs from the dataloader
            input_ids, attention_mask, labels = [x.to(device) for x in batch]

            # Clear previously calculated gradients
            optimizer.zero_grad()

            # Forward pass: Get the model outputs
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            # Compute the loss and accumulate it
            loss = outputs.loss
            total_loss += loss.item()

            # Backpropagation
            loss.backward()

            # Gradient clipping (recommended for BERT fine-tuning)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Training Loss: {avg_train_loss:.3f}")

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_accuracy = 0
        val_loss = 0
        nb_val_steps = 0

        # Disable gradient calculation for validation (saves memory and computation)
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, attention_mask, labels = [x.to(device) for x in batch]

                # Forward pass
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

                # Get predictions
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).flatten()

                # Calculate accuracy
                val_accuracy += (preds == labels).cpu().numpy().mean()
                nb_val_steps += 1

        avg_val_accuracy = val_accuracy / nb_val_steps
        avg_val_loss = val_loss / nb_val_steps

        print(f"Validation Loss: {avg_val_loss:.3f}, Validation Accuracy: {avg_val_accuracy:.3f}")



# Set the number of training epochs
epochs = 3

# Train the model
train(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs)


# Save the trained model and tokenizer
model_save_path = "bert_sentiment_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model and tokenizer saved to {model_save_path}.")

