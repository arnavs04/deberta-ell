import os
import random
import gc
import time
import math
from pathlib import Path
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

import torch
from torch import nn

import numpy as np
from sklearn.metrics import mean_squared_error

from ptflops import get_model_complexity_info


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mcrmse(y_trues, y_preds):
    mse = np.mean((y_trues - y_preds)**2, axis=0)
    rmse = np.sqrt(mse)
    return np.mean(rmse), rmse.tolist()


def get_score(y_trues, y_preds):
    # Ensure inputs are numpy arrays
    y_trues = np.array(y_trues)
    y_preds = np.array(y_preds)
    return mcrmse(y_trues, y_preds)


def get_logger(filename='./train'):
    logger = getLogger(__name__)
    if not logger.hasHandlers():
        logger.setLevel(INFO)
        stream_handler = StreamHandler()
        stream_handler.setFormatter(Formatter("%(message)s"))

        file_handler = FileHandler(filename=f"{filename}.log")
        file_handler.setFormatter(Formatter("%(message)s"))

        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    return logger


def as_minutes(s):
    m = math.floor(s / 60)
    return f'{m}m {s - m * 60}s'


def time_since(since, percent):
    now = time.time()
    elapsed = now - since
    est_total = elapsed / percent
    remain = est_total - elapsed
    return f'{as_minutes(elapsed)} (remain {as_minutes(remain)})'


def save_model(model: nn.Module, target_dir: str, model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    
    model_save_path = target_dir_path / model_name
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def is_torch_available():
    return torch is not None


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_in_bytes = total_params * 4
    size_in_mb = size_in_bytes / (1024 ** 2)
    return size_in_mb


def clear_model_from_memory(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def count_model_flops(model: nn.Module, input_size=(3, 224, 224), print_results=True):
    try:
        macs, params = get_model_complexity_info(
            model, input_size, as_strings=False, print_per_layer_stat=False, verbose=False
        )
        
        gflops = macs / 1e9  # convert MACs to GFLOPs

        if print_results:
            print(f'Computational Complexity: {gflops:.3f} GFLOPs')
            print(f'Number of Parameters: {params / 1e6:.3f} M')

        return gflops, params
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None