import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from utils import *

# LOGGER = get_logger()


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True) if config_path is None else torch.load(config_path)
        if not config_path:
            self.config.update({
                "hidden_dropout": 0.0,
                "hidden_dropout_prob": 0.0,
                "attention_dropout": 0.0,
                "attention_probs_dropout_prob": 0.0
            })
        
        self.model = AutoModel.from_pretrained(cfg.model, config=self.config) if pretrained else AutoModel(self.config)
        if cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, 6)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        last_hidden_states = self.model(**inputs)[0]
        return self.pool(last_hidden_states, inputs['attention_mask'])

    def forward(self, inputs):
        return self.fc(self.feature(inputs))

    # def forward(self, input_ids, attention_mask):
    #     # for checking model complexity and size
    #     inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    #     feature = self.feature(inputs)
    #     output = self.fc(feature)
    #     return output