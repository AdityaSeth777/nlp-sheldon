# utils/data_preprocessing.py
import json

def load_data(file_path):
    with open(file_path, 'r') as file:
        dialogues = json.load(file)
    return dialogues


def preprocess_dialogue(dialogues, tokenizer):
    input_texts = [d["input"] for d in dialogues]
    target_texts = [d["target"] for d in dialogues]

    input_encodings = tokenizer(
        input_texts, padding=True, truncation=True, return_tensors="tf", add_special_tokens=True)
    target_encodings = tokenizer(
        target_texts, padding=True, truncation=True, return_tensors="tf", add_special_tokens=True)

    return input_encodings, target_encodings
