from src.configs import *
from src.model_builder import *
from src.utils import count_model_complexity, model_size_mb

model = CustomModel(CFG, pretrained=True)
model_size_mb(model)
gflops, params = count_model_complexity(model, seq_length=256, vocab_size=30522)