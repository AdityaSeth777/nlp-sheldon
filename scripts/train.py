# scripts/train.py
import sys
import os
import repackage
import torch
from torch.optim import AdamW
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
repackage.up()
from utils.data_preprocessing import load_data, preprocess_dialogue
repackage.up()

print("Current working directory:", os.getcwd())
# Add the parent directory to the path to resolve imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Load data
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data', 'dialogues.json')
dialogues = load_data(data_path)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Preprocess data
input_encodings, target_encodings = preprocess_dialogue(dialogues, tokenizer)

# Print shapes for debugging
print("Input IDs shape:", input_encodings["input_ids"].shape)
print("Attention mask shape:", input_encodings["attention_mask"].shape)
print("Target IDs shape:", target_encodings["input_ids"].shape)

# Create a custom PyTorch Dataset
class DialogueDataset(Dataset):
    def __init__(self, input_encodings, target_encodings):
        self.input_encodings = input_encodings
        self.target_encodings = target_encodings

    def __len__(self):
        return len(self.input_encodings["input_ids"])

    def __getitem__(self, idx):
        input_ids = self.input_encodings["input_ids"][idx].numpy().tolist()
        attention_mask = self.input_encodings["attention_mask"][idx].numpy().tolist()
        labels = self.target_encodings["input_ids"][idx].numpy().tolist()
        
        # Convert lists to PyTorch tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        # Replace padding token ID with -100 for loss computation
        pad_token_id = tokenizer.pad_token_id
        labels = torch.where(labels == pad_token_id, torch.tensor(-100), labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Create DataLoader
dataset = DialogueDataset(input_encodings, target_encodings)
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(10):
    print(f"Epoch {epoch+1}/10")
    for batch_idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Ensure that labels are properly formatted and of the correct shape
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: Loss = {loss.item()}")

# Save model
model.save_pretrained('models/t5_model')
tokenizer.save_pretrained('models/t5_tokenizer')