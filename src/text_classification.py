import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Check if PyTorch is using the GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("BERT Tokenizer loaded successfully.")
