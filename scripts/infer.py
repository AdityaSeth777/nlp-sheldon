import sys
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Add the parent directory to the path to resolve imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.data_preprocessing import preprocess_dialogue  # Import preprocess function if needed

# Load model and tokenizer
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'models', 't5_model')
tokenizer_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'models', 't5_tokenizer')

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_response(input_text):
    model.eval()

    # Encode input text
    inputs = tokenizer.encode(input_text, return_tensors='pt', truncation=True, padding=True).to(device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=50, num_beams=5, early_stopping=True)

    # Decode output text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage
if __name__ == "__main__":
    input_text = input("Enter your input text: ")
    response = generate_response(input_text)
    print("Input:", input_text)
    print("Response:", response)


# Example usage
#input_text = "If Batman were bitten by a radioactive Man-Bat, and then fought crime disguised as Man-Bat, would he be Man-Bat-Man-Bat-Man or simply Man-Bat-Man-Bat-Batman?"
#response = generate_response(input_text, tokenizer, model)
#print(response)
