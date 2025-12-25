# app.py
import os
import re
import pickle
import numpy as np
import torch
import torch.nn as nn

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler

# =========================================================
# 0. 基本設定（與訓練時一致）
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "roberta-base"
MAX_LEN = 512

MODEL_PATH = "best_model_state_local.bin"
SCALER_PATH = "scaler_local.pkl"

print(f"Using device: {DEVICE}")

# =========================================================
# 1. Aspect 與正則（完全沿用）
# =========================================================
ASPECTS = {
    "plot": ["plot", "story", "writing", "pacing", "narrative", "ending", "twist", "boring"],
    "character": ["character", "protagonist", "villain", "cast", "development", "waifu", "personality"],
    "animation": ["animation", "art", "visual", "design", "style", "cgi", "quality"],
    "music": ["music", "soundtrack", "ost", "bgm", "opening", "ending", "song", "voice"],
    "voice": ["voice", "dub", "sub", "acting", "seiyuu"],
    "adaptation": ["adaptation", "manga", "novel", "source", "faithful", "filler"]
}

ASPECT_PATTERNS = {
    k: re.compile(r"\b(" + "|".join(re.escape(w) for w in v) + r")\b", re.IGNORECASE)
    for k, v in ASPECTS.items()
}

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

N_NUMERIC_FEATURES = len(ASPECTS) * 4  # mean, min, max, count

analyzer = SentimentIntensityAnalyzer()

# =========================================================
# 2. 特徵萃取（完全複製你的原邏輯）
# =========================================================
def extract_aspect_features(text: str) -> np.ndarray:
    sents = SENT_SPLIT.split(text)
    feats = []

    for aspect, pat in ASPECT_PATTERNS.items():
        mentions = 0
        scores = []

        for s in sents:
            found = pat.findall(s)
            if found:
                mentions += len(found)
                scores.append(analyzer.polarity_scores(s)["compound"])

        if scores:
            feats.extend([
                np.mean(scores),
                np.min(scores),
                np.max(scores),
                float(mentions)
            ])
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0])

    return np.array(feats, dtype=np.float32)

# =========================================================
# 3. 模型架構（完全一致）
# =========================================================
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
        combined = torch.cat((cls, numeric_feats), dim=1)
        return self.regressor(combined)

# =========================================================
# 4. FastAPI 初始化
# =========================================================
app = FastAPI(title="Anime Review Score API")

class InputText(BaseModel):
    text: str

# =========================================================
# 5. 載入 Tokenizer / Model / Scaler（啟動時）
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = RobertaWithTabular(n_numeric_features=N_NUMERIC_FEATURES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

with open(SCALER_PATH, "rb") as f:
    scaler: StandardScaler = pickle.load(f)

print("Model & Scaler loaded successfully.")

# =========================================================
# 6. API Endpoint
# =========================================================
@app.post("/predict")
def predict(data: InputText):
    text = data.text.strip()
    if not text:
        return {"error": "Empty input text."}

    # ---- tokenizer ----
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    # ---- numeric features ----
    numeric_feats = extract_aspect_features(text)
    numeric_feats = scaler.transform([numeric_feats])

    # ---- model inference ----
    with torch.no_grad():
        output = model(
            encoding["input_ids"].to(DEVICE),
            encoding["attention_mask"].to(DEVICE),
            torch.tensor(numeric_feats, dtype=torch.float).to(DEVICE)
        )

    score = float(output.item())
    score = float(np.clip(score, 1.0, 10.0))

    return {
        "score": round(score, 2)
    }
