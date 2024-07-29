import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from utils import *
from configs import *
from data_setup import *
from engine import *
from model_builder import *
from metrics import *

warnings.filterwarnings("ignore")
print("Imports Done")

# set environment variable
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
OUTPUT_DIR = './'

# setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOGGER = get_logger()
seed_everything(seed=42)

# load data
train = pd.read_csv('../deberta-ell/data/feedback-prize-english-language-learning/train.csv')
test = pd.read_csv('../deberta-ell/data/feedback-prize-english-language-learning/test.csv')

# prepare data
Fold = MultilabelStratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for n, (train_index, val_index) in enumerate(Fold.split(train, train[CFG.target_cols])):
    train.loc[val_index, 'fold'] = int(n)
train['fold'] = train['fold'].astype(int)

if CFG.debug:
    train = train.sample(n=1000, random_state=0).reset_index(drop=True)

# setup tokenizer
tokenizer = AutoTokenizer.from_pretrained(CFG.model)
tokenizer.save_pretrained('./' + 'tokenizer/')
CFG.tokenizer = tokenizer

# get max length
lengths = []
tk0 = tqdm(train['full_text'].fillna("").values, total=len(train))
for text in tk0:
    length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
    lengths.append(length)
CFG.max_len = max(lengths) + 3  # cls & sep & sep

# helper functions
def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.model.named_parameters() 
                    if not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.model.named_parameters() 
                    if any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if "model" not in n],
         'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters

def get_scheduler(cfg, optimizer, num_train_steps):
    if cfg.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps,
            num_training_steps=num_train_steps
        )
    elif cfg.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps,
            num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
        )
    return scheduler

# training loop
def train_loop(folds, fold):
    LOGGER.info(f"-------------- Fold: {fold} Training --------------")

    # prepare data
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    valid_labels = valid_folds[CFG.target_cols].values
    
    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size * 2,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # model
    model = CustomModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, OUTPUT_DIR + 'config.pth')
    model.to(device)
    
    # optimizer and scheduler
    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=CFG.encoder_lr,
        decoder_lr=CFG.decoder_lr,
        weight_decay=CFG.weight_decay
    )
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)
    
    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # loop
    criterion = nn.SmoothL1Loss(reduction='mean')
    best_score = np.inf

    for epoch in range(CFG.epochs):
        start_time = time.time()

        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device)
        
        # scoring
        score, scores = get_score(valid_labels, predictions)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}  Scores: {scores}')
        
        if best_score > score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                       OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")

    predictions = torch.load(OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                             map_location=torch.device('cpu'))['predictions']
    valid_folds[[f"pred_{c}" for c in CFG.target_cols]] = predictions

    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_folds

def get_result(oof_df):
    labels = oof_df[CFG.target_cols].values
    preds = oof_df[[f"pred_{c}" for c in CFG.target_cols]].values
    score, scores = get_score(labels, preds)
    LOGGER.info(f'Score: {score:<.4f}  Scores: {scores}')

if CFG.train:
    oof_df = pd.DataFrame()
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            _oof_df = train_loop(train, fold)
            oof_df = pd.concat([oof_df, _oof_df])
            LOGGER.info(f"-------------- Fold: {fold} Result --------------")
            get_result(_oof_df)
    oof_df = oof_df.reset_index(drop=True)
    LOGGER.info(f"-------------- CV --------------")
    get_result(oof_df)
    oof_df.to_pickle(OUTPUT_DIR + 'oof_df.pkl')