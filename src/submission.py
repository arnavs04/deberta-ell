import pandas as pd
import torch
from transformers import AutoTokenizer
from model_builder import CustomModel
from configs import CFG

OUTPUT_DIR = '../deberta-ell/logs'
MODEL_PATH = '../deberta-ell/models/params/'

def load_model(model_path):
    model = CustomModel(CFG, pretrained=True)
    model.load_state_dict(torch.load(model_path))  # Load the model weights from the .pth file
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    return model, tokenizer

def prepare_input(text, tokenizer, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
    return inputs

def infer(model, tokenizer, text):
    inputs = prepare_input(text, tokenizer)
    with torch.no_grad():
        outputs = model(**inputs)  # Make sure to unpack the inputs correctly
    return outputs

def main():
    model_path = MODEL_PATH
    model, tokenizer = load_model(model_path)
    
    test_df = pd.read_csv("../deberta-ell/data/feedback-prize-english-language-learning/test.csv")
    
    submission_df = pd.DataFrame()
    submission_df['text_id'] = test_df['text_id']

    score_columns = ['pred_cohesion', 'pred_syntax', 'pred_vocabulary', 'pred_phraseology', 'pred_grammar', 'pred_conventions']
    for col in score_columns:
        submission_df[col] = 0.0
    
    for idx, row in test_df.iterrows():
        text = row['full_text']
        outputs = infer(model, tokenizer, text)
        
        outputs = outputs.squeeze().numpy()
        
        submission_df.loc[idx, score_columns] = outputs
    
    submission_df.to_csv("submission.csv", index=False, path_or_buf=OUTPUT_DIR)
    print("Predictions saved to submission.csv")

if __name__ == "__main__":
    main()