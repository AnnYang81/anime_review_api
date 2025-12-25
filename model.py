# model.py
import torch
import torch.nn as nn
from transformers import AutoModel

MODEL_NAME = "roberta-base"

class RobertaWithTabular(nn.Module):
    def __init__(self, n_numeric_features):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.roberta.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden + n_numeric_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, input_ids, attention_mask, numeric_feats):
        out = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = out.last_hidden_state[:, 0, :]
        x = torch.cat([cls, numeric_feats], dim=1)
        return self.regressor(x)
