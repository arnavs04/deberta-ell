import torch
from transformers import AutoTokenizer
from model_builder import CustomModel
from configs import CFG

def load_model():
    model = CustomModel(CFG, pretrained=True)
    model.eval() 
    tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    return model, tokenizer

def prepare_input(text, tokenizer, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    return inputs

def infer(model, tokenizer, text):
    inputs = prepare_input(text, tokenizer)
    with torch.no_grad():  
        outputs = model(inputs)
    return outputs

def main():
    model, tokenizer = load_model()
    text = "This is a sample input for the CustomModel."
    outputs = infer(model, tokenizer, text)
    print("Raw output scores:", outputs)

if __name__ == "__main__":
    main()