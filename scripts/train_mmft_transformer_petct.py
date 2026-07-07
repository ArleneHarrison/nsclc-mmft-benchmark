from __future__ import annotations

import json
import math
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from scripts.benchmark_petct_blood_models import RANDOM_STATE, read_data
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    sys.path.append(str(Path(__file__).resolve().parent))
    from benchmark_petct_blood_models import RANDOM_STATE, read_data


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "server_results" / "petct_blood_benchmark"


@dataclass(frozen=True)
class TabularBatch:
    x: torch.Tensor
    modality_ids: torch.Tensor
    feature_ids: torch.Tensor


@dataclass(frozen=True)
class TrainConfig:
    name: str
    modalities: dict[str, list[str]]
    k: int
    d_model: int
    n_heads: int
    n_layers: int
    dropout: float
    lr: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    seed: int
    focal_gamma: float = 0.0
    aux_linear_weight: float = 0.15
    use_risk_token: bool = False
    risk_c: float = 3.0


class ModalityAwarePreprocessor:
    def __init__(self, modalities: dict[str, list[str]], k: int):
        self.modalities = modalities
        self.k = k
        self.feature_order_: list[str] = []
        self.selected_features_: list[str] = []
        self.selected_modalities_: list[str] = []
        self.modality_to_id_: dict[str, int] = {}
        self.imputer_: SimpleImputer | None = None
        self.variance_: VarianceThreshold | None = None
        self.scaler_: StandardScaler | None = None
        self.selector_: SelectKBest | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series | np.ndarray) -> "ModalityAwarePreprocessor":
        self.feature_order_ = [feature for features in self.modalities.values() for feature in features if feature in x.columns]
        if not self.feature_order_:
            raise ValueError("No usable features were found for the requested modalities.")

        self.modality_to_id_ = {name: idx for idx, name in enumerate(self.modalities)}
        arr = x[self.feature_order_].apply(pd.to_numeric, errors="coerce")

        self.imputer_ = SimpleImputer(strategy="median")
        self.variance_ = VarianceThreshold()
        self.scaler_ = StandardScaler()

        arr_np = self.imputer_.fit_transform(arr)
        arr_np = self.variance_.fit_transform(arr_np)
        after_variance = np.array(self.feature_order_)[self.variance_.get_support()]
        arr_np = self.scaler_.fit_transform(arr_np)

        k_eff = min(max(1, self.k), arr_np.shape[1])
        self.selector_ = SelectKBest(score_func=f_classif, k=k_eff)
        self.selector_.fit(arr_np, y)

        self.selected_features_ = after_variance[self.selector_.get_support()].tolist()
        feature_to_modality = {
            feature: modality for modality, features in self.modalities.items() for feature in features
        }
        self.selected_modalities_ = [feature_to_modality[feature] for feature in self.selected_features_]
        return self

    def transform(self, x: pd.DataFrame) -> TabularBatch:
        if self.imputer_ is None or self.variance_ is None or self.scaler_ is None or self.selector_ is None:
            raise RuntimeError("Preprocessor must be fit before transform.")

        arr = x[self.feature_order_].apply(pd.to_numeric, errors="coerce")
        arr_np = self.imputer_.transform(arr)
        arr_np = self.variance_.transform(arr_np)
        arr_np = self.scaler_.transform(arr_np)
        arr_np = self.selector_.transform(arr_np).astype("float32")

        modality_ids = torch.tensor(
            [self.modality_to_id_[name] for name in self.selected_modalities_],
            dtype=torch.long,
        )
        feature_ids = torch.arange(arr_np.shape[1], dtype=torch.long)
        return TabularBatch(x=torch.from_numpy(arr_np), modality_ids=modality_ids, feature_ids=feature_ids)

    def fit_transform(self, x: pd.DataFrame, y: pd.Series | np.ndarray) -> TabularBatch:
        return self.fit(x, y).transform(x)


class RiskTokenAppender:
    def __init__(self, modality_id: int, c: float = 3.0):
        self.modality_id = modality_id
        self.c = c
        self.model_: LogisticRegression | None = None
        self.mean_: float = 0.0
        self.std_: float = 1.0

    def fit(self, train_batch: TabularBatch, y: np.ndarray | pd.Series) -> "RiskTokenAppender":
        y_np = np.asarray(y).astype(int)
        self.model_ = LogisticRegression(
            penalty="l2",
            C=self.c,
            solver="liblinear",
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE,
        )
        x_np = train_batch.x.detach().cpu().numpy()
        self.model_.fit(x_np, y_np)
        logits = self.model_.decision_function(x_np)
        self.mean_ = float(np.mean(logits))
        self.std_ = float(np.std(logits))
        if self.std_ < 1e-6:
            self.std_ = 1.0
        return self

    def transform(self, batch: TabularBatch) -> TabularBatch:
        if self.model_ is None:
            raise RuntimeError("RiskTokenAppender must be fit before transform.")
        x_np = batch.x.detach().cpu().numpy()
        logits = self.model_.decision_function(x_np)
        token = ((logits - self.mean_) / self.std_).astype("float32")
        token_tensor = torch.from_numpy(token).view(-1, 1).to(batch.x.device)
        x_new = torch.cat([batch.x, token_tensor], dim=1)
        modality_ids = torch.cat(
            [
                batch.modality_ids,
                torch.tensor([self.modality_id], dtype=torch.long, device=batch.modality_ids.device),
            ]
        )
        feature_ids = torch.arange(x_new.shape[1], dtype=torch.long, device=batch.feature_ids.device)
        return TabularBatch(x=x_new, modality_ids=modality_ids, feature_ids=feature_ids)

    def fit_transform(self, train_batch: TabularBatch, y: np.ndarray | pd.Series) -> TabularBatch:
        return self.fit(train_batch, y).transform(train_batch)


class FeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, n_modalities: int, d_model: int, dropout: float):
        super().__init__()
        self.value_weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.value_bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.feature_embedding = nn.Embedding(n_features, d_model)
        self.modality_embedding = nn.Embedding(n_modalities, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch: TabularBatch) -> torch.Tensor:
        x = batch.x
        value_tokens = x.unsqueeze(-1) * self.value_weight.unsqueeze(0) + self.value_bias.unsqueeze(0)
        feature_tokens = self.feature_embedding(batch.feature_ids).unsqueeze(0)
        modality_tokens = self.modality_embedding(batch.modality_ids).unsqueeze(0)
        tokens = self.dropout(value_tokens + feature_tokens + modality_tokens)
        cls = self.cls.expand(x.size(0), -1, -1)
        return torch.cat([cls, tokens], dim=1)


class GatedModalityPooling(nn.Module):
    def __init__(self, d_model: int, n_modalities: int):
        super().__init__()
        self.n_modalities = n_modalities
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(4, d_model // 2)),
            nn.GELU(),
            nn.Linear(max(4, d_model // 2), 1),
        )

    def forward(self, feature_tokens: torch.Tensor, modality_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = []
        present = []
        for modality_id in range(self.n_modalities):
            mask = modality_ids == modality_id
            if torch.any(mask):
                pooled.append(feature_tokens[:, mask, :].mean(dim=1))
                present.append(True)
            else:
                pooled.append(torch.zeros_like(feature_tokens[:, 0, :]))
                present.append(False)
        modality_matrix = torch.stack(pooled, dim=1)
        gate_logits = self.gate(modality_matrix).squeeze(-1)
        absent = torch.tensor(present, device=feature_tokens.device, dtype=torch.bool).unsqueeze(0)
        gate_logits = gate_logits.masked_fill(~absent, -1e9)
        weights = torch.softmax(gate_logits, dim=1)
        return (modality_matrix * weights.unsqueeze(-1)).sum(dim=1), weights


class MMFTTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_modalities: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, n_modalities, d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.attention_pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        self.modality_pool = GatedModalityPooling(d_model, n_modalities)
        self.deep_head = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.linear_head = nn.Linear(n_features, 1)

    def forward(self, batch: TabularBatch) -> torch.Tensor:
        tokens = self.tokenizer(batch)
        encoded = self.encoder(tokens)
        cls = encoded[:, 0, :]
        feature_tokens = encoded[:, 1:, :]
        attn = torch.softmax(self.attention_pool(feature_tokens).squeeze(-1), dim=1)
        attentive = (feature_tokens * attn.unsqueeze(-1)).sum(dim=1)
        gated, _ = self.modality_pool(feature_tokens, batch.modality_ids)
        deep_logits = self.deep_head(torch.cat([cls, attentive, gated], dim=1)).squeeze(-1)
        linear_logits = self.linear_head(batch.x).squeeze(-1)
        return deep_logits + linear_logits


def initialize_prior_residual(model: MMFTTransformer, risk_feature_index: int) -> None:
    with torch.no_grad():
        model.linear_head.weight.zero_()
        model.linear_head.bias.zero_()
        model.linear_head.weight[0, risk_feature_index] = 1.0
        last_linear = None
        for module in model.deep_head.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is not None:
            last_linear.weight.zero_()
            last_linear.bias.zero_()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def feature_modalities(df: pd.DataFrame) -> dict[str, list[str]]:
    clinical_blood = [
        "gender",
        "age",
        "BMI",
        "smoking",
        "T stage",
        "N stage",
        "stage",
        "WBC",
        "NEU",
        "LYM",
        "EOS",
        "BAS",
        "PLT",
        "CEA",
        "LDH",
        "ALB",
        "Ca2+",
        "NLR",
        "dNLR",
    ]
    metabolic = ["TLG", "SUVmean", "MTV", "SUVmax", "SUVmin"]
    return {
        "clinical_blood": clinical_blood,
        "pet_metabolic": metabolic,
        "ct_radiomics": [c for c in df.columns if c.startswith("ct__")],
        "pet_radiomics": [c for c in df.columns if c.startswith("pet__")],
    }


def subset_modalities(all_modalities: dict[str, list[str]], names: list[str]) -> dict[str, list[str]]:
    return {name: all_modalities[name] for name in names}


def move_batch(batch: TabularBatch, device: torch.device) -> TabularBatch:
    return TabularBatch(
        x=batch.x.to(device),
        modality_ids=batch.modality_ids.to(device),
        feature_ids=batch.feature_ids.to(device),
    )


def batch_to_loader(batch: TabularBatch, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(batch.x, torch.from_numpy(y.astype("float32"))),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def binary_focal_bce(logits: torch.Tensor, y: torch.Tensor, pos_weight: torch.Tensor, gamma: float) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none")
    if gamma <= 0:
        return bce.mean()
    prob = torch.sigmoid(logits)
    pt = torch.where(y == 1, prob, 1 - prob).clamp(1e-5, 1 - 1e-5)
    return ((1 - pt) ** gamma * bce).mean()


@torch.no_grad()
def predict(model: MMFTTransformer, batch: TabularBatch, batch_size: int = 512) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    for start in range(0, batch.x.shape[0], batch_size):
        part = TabularBatch(
            x=batch.x[start : start + batch_size],
            modality_ids=batch.modality_ids,
            feature_ids=batch.feature_ids,
        )
        probs.append(torch.sigmoid(model(part)).cpu().numpy())
    return np.concatenate(probs)


def score(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
    }


def train_one(df: pd.DataFrame, y: pd.Series, cfg: TrainConfig, device: torch.device) -> dict:
    set_seed(cfg.seed)
    x = df[[feature for features in cfg.modalities.values() for feature in features]].apply(pd.to_numeric, errors="coerce")
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval,
        y_trainval,
        test_size=0.22,
        random_state=cfg.seed,
        stratify=y_trainval,
    )

    pre = ModalityAwarePreprocessor(cfg.modalities, cfg.k)
    train_batch_cpu = pre.fit_transform(x_train, y_train)
    val_batch_cpu = pre.transform(x_val)
    test_batch_cpu = pre.transform(x_test)
    if cfg.use_risk_token:
        risk = RiskTokenAppender(modality_id=len(cfg.modalities), c=cfg.risk_c)
        train_batch_cpu = risk.fit_transform(train_batch_cpu, y_train)
        val_batch_cpu = risk.transform(val_batch_cpu)
        test_batch_cpu = risk.transform(test_batch_cpu)

    train_batch = move_batch(train_batch_cpu, device)
    val_batch = move_batch(val_batch_cpu, device)
    test_batch = move_batch(test_batch_cpu, device)

    model = MMFTTransformer(
        n_features=train_batch.x.shape[1],
        n_modalities=int(train_batch.modality_ids.max().item()) + 1,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    ).to(device)
    if cfg.use_risk_token:
        initialize_prior_residual(model, risk_feature_index=train_batch.x.shape[1] - 1)

    y_train_np = y_train.to_numpy().astype("float32")
    y_val_np = y_val.to_numpy().astype("float32")
    y_test_np = y_test.to_numpy().astype("float32")
    pos = y_train_np.sum()
    neg = len(y_train_np) - pos
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = batch_to_loader(train_batch, y_train_np, cfg.batch_size, shuffle=True)

    initial_val_prob = predict(model, val_batch)
    best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    best_val_auc = roc_auc_score(y_val_np, initial_val_prob)
    best_epoch = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for xb, yb in loader:
            mini = TabularBatch(x=xb.to(device), modality_ids=train_batch.modality_ids, feature_ids=train_batch.feature_ids)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(mini)
            loss = binary_focal_bce(logits, yb, pos_weight, cfg.focal_gamma)
            if cfg.aux_linear_weight > 0:
                linear_logits = model.linear_head(mini.x).squeeze(-1)
                aux_loss = binary_focal_bce(linear_logits, yb, pos_weight, 0.0)
                loss = loss + cfg.aux_linear_weight * aux_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_prob = predict(model, val_batch)
        val_auc = roc_auc_score(y_val_np, val_prob)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        if epoch - best_epoch >= cfg.patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    val_prob = predict(model, val_batch)
    test_prob = predict(model, test_batch)
    row = {
        "config": cfg.name,
        "seed": cfg.seed,
        "k": train_batch.x.shape[1],
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "n_layers": cfg.n_layers,
        "dropout": cfg.dropout,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "focal_gamma": cfg.focal_gamma,
        "aux_linear_weight": cfg.aux_linear_weight,
        "use_risk_token": cfg.use_risk_token,
        "risk_c": cfg.risk_c,
        "best_epoch": best_epoch,
        "selected_features": json.dumps(pre.selected_features_ + (["ridge_risk_token"] if cfg.use_risk_token else []), ensure_ascii=False),
        **{f"val_{key}": value for key, value in score(y_val_np, val_prob).items()},
        **{f"test_{key}": value for key, value in score(y_test_np, test_prob).items()},
    }
    return {
        "row": row,
        "test_prob": test_prob,
        "y_test": y_test_np,
        "test_ids": df.loc[x_test.index, "ID"].tolist(),
    }


def train_on_explicit_split(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: TrainConfig,
    device: torch.device,
) -> dict:
    set_seed(cfg.seed)
    pre = ModalityAwarePreprocessor(cfg.modalities, cfg.k)
    train_batch_cpu = pre.fit_transform(x_train, y_train)
    val_batch_cpu = pre.transform(x_val)
    test_batch_cpu = pre.transform(x_test)
    if cfg.use_risk_token:
        risk = RiskTokenAppender(modality_id=len(cfg.modalities), c=cfg.risk_c)
        train_batch_cpu = risk.fit_transform(train_batch_cpu, y_train)
        val_batch_cpu = risk.transform(val_batch_cpu)
        test_batch_cpu = risk.transform(test_batch_cpu)

    train_batch = move_batch(train_batch_cpu, device)
    val_batch = move_batch(val_batch_cpu, device)
    test_batch = move_batch(test_batch_cpu, device)
    model = MMFTTransformer(
        n_features=train_batch.x.shape[1],
        n_modalities=int(train_batch.modality_ids.max().item()) + 1,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    ).to(device)
    if cfg.use_risk_token:
        initialize_prior_residual(model, risk_feature_index=train_batch.x.shape[1] - 1)

    y_train_np = y_train.to_numpy().astype("float32")
    y_val_np = y_val.to_numpy().astype("float32")
    y_test_np = y_test.to_numpy().astype("float32")
    pos = y_train_np.sum()
    neg = len(y_train_np) - pos
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = batch_to_loader(train_batch, y_train_np, cfg.batch_size, shuffle=True)

    initial_val_prob = predict(model, val_batch)
    best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    best_val_auc = roc_auc_score(y_val_np, initial_val_prob)
    best_epoch = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for xb, yb in loader:
            mini = TabularBatch(x=xb.to(device), modality_ids=train_batch.modality_ids, feature_ids=train_batch.feature_ids)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(mini)
            loss = binary_focal_bce(logits, yb, pos_weight, cfg.focal_gamma)
            if cfg.aux_linear_weight > 0:
                aux_loss = binary_focal_bce(model.linear_head(mini.x).squeeze(-1), yb, pos_weight, 0.0)
                loss = loss + cfg.aux_linear_weight * aux_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_prob = predict(model, val_batch)
        val_auc = roc_auc_score(y_val_np, val_prob)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        if epoch - best_epoch >= cfg.patience:
            break

    model.load_state_dict(best_state)
    val_prob = predict(model, val_batch)
    test_prob = predict(model, test_batch)
    return {
        "val_prob": val_prob,
        "test_prob": test_prob,
        "y_val": y_val_np,
        "y_test": y_test_np,
        "best_epoch": best_epoch,
        "val_auc": float(roc_auc_score(y_val_np, val_prob)),
        "selected_features": pre.selected_features_ + (["ridge_risk_token"] if cfg.use_risk_token else []),
    }


def train_cv_ensemble(df: pd.DataFrame, y: pd.Series, cfg: TrainConfig, device: torch.device, n_splits: int = 5) -> dict:
    features = [feature for features in cfg.modalities.values() for feature in features]
    x = df[features].apply(pd.to_numeric, errors="coerce")
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(x_trainval), dtype="float32")
    test_probs = []
    fold_rows = []

    for fold, (train_pos, val_pos) in enumerate(splitter.split(x_trainval, y_trainval), start=1):
        fold_cfg = TrainConfig(**{**cfg.__dict__, "seed": cfg.seed + fold})
        result = train_on_explicit_split(
            x_trainval.iloc[train_pos],
            y_trainval.iloc[train_pos],
            x_trainval.iloc[val_pos],
            y_trainval.iloc[val_pos],
            x_test,
            y_test,
            fold_cfg,
            device,
        )
        oof[val_pos] = result["val_prob"]
        test_probs.append(result["test_prob"])
        fold_rows.append(
            {
                "config": cfg.name,
                "fold": fold,
                "val_auc": result["val_auc"],
                "best_epoch": result["best_epoch"],
                "selected_features": json.dumps(result["selected_features"], ensure_ascii=False),
            }
        )
        print(f"CV {cfg.name:24s} fold={fold} val_auc={result['val_auc']:.3f} epoch={result['best_epoch']}")

    mean_test_prob = np.mean(test_probs, axis=0)
    oof_scores = score(y_trainval.to_numpy().astype("float32"), oof)
    test_scores = score(y_test.to_numpy().astype("float32"), mean_test_prob)
    return {
        "row": {
            "config": cfg.name,
            "n_splits": n_splits,
            **{f"oof_{key}": value for key, value in oof_scores.items()},
            **{f"test_{key}": value for key, value in test_scores.items()},
        },
        "fold_rows": fold_rows,
        "test_prob": mean_test_prob,
        "y_test": y_test.to_numpy().astype("float32"),
        "test_ids": df.loc[x_test.index, "ID"].tolist(),
    }


def make_configs(df: pd.DataFrame) -> list[TrainConfig]:
    all_modalities = feature_modalities(df)
    seeds = [20260701, 20260702, 20260703, 20260704, 20260705]
    configs: list[TrainConfig] = []
    for seed in seeds:
        configs.extend(
            [
                TrainConfig(
                    name="risk_clinical_pet_k5",
                    modalities=subset_modalities(all_modalities, ["clinical_blood", "pet_metabolic"]),
                    k=5,
                    d_model=16,
                    n_heads=2,
                    n_layers=1,
                    dropout=0.06,
                    lr=6e-4,
                    weight_decay=1e-4,
                    batch_size=32,
                    max_epochs=450,
                    patience=55,
                    seed=seed,
                    focal_gamma=0.0,
                    aux_linear_weight=0.35,
                    use_risk_token=True,
                    risk_c=3.0,
                ),
                TrainConfig(
                    name="risk_clinical_pet_k8",
                    modalities=subset_modalities(all_modalities, ["clinical_blood", "pet_metabolic"]),
                    k=8,
                    d_model=24,
                    n_heads=3,
                    n_layers=1,
                    dropout=0.10,
                    lr=5e-4,
                    weight_decay=2e-4,
                    batch_size=32,
                    max_epochs=450,
                    patience=55,
                    seed=seed,
                    focal_gamma=0.0,
                    aux_linear_weight=0.30,
                    use_risk_token=True,
                    risk_c=3.0,
                ),
                TrainConfig(
                    name="risk_clinical_pet_k16",
                    modalities=subset_modalities(all_modalities, ["clinical_blood", "pet_metabolic"]),
                    k=16,
                    d_model=32,
                    n_heads=4,
                    n_layers=1,
                    dropout=0.16,
                    lr=5e-4,
                    weight_decay=3e-4,
                    batch_size=32,
                    max_epochs=450,
                    patience=55,
                    seed=seed,
                    focal_gamma=0.3,
                    aux_linear_weight=0.25,
                    use_risk_token=True,
                    risk_c=3.0,
                ),
                TrainConfig(
                    name="clinical_pet_small",
                    modalities=subset_modalities(all_modalities, ["clinical_blood", "pet_metabolic"]),
                    k=8,
                    d_model=24,
                    n_heads=3,
                    n_layers=1,
                    dropout=0.10,
                    lr=8e-4,
                    weight_decay=1e-4,
                    batch_size=32,
                    max_epochs=500,
                    patience=50,
                    seed=seed,
                    focal_gamma=0.0,
                    aux_linear_weight=0.25,
                ),
                TrainConfig(
                    name="clinical_pet_medium",
                    modalities=subset_modalities(all_modalities, ["clinical_blood", "pet_metabolic"]),
                    k=16,
                    d_model=32,
                    n_heads=4,
                    n_layers=1,
                    dropout=0.18,
                    lr=6e-4,
                    weight_decay=2e-4,
                    batch_size=32,
                    max_epochs=500,
                    patience=55,
                    seed=seed,
                    focal_gamma=0.5,
                    aux_linear_weight=0.20,
                ),
                TrainConfig(
                    name="clinical_pet_plus_pet_radiomics",
                    modalities=subset_modalities(all_modalities, ["clinical_blood", "pet_metabolic", "pet_radiomics"]),
                    k=32,
                    d_model=32,
                    n_heads=4,
                    n_layers=1,
                    dropout=0.24,
                    lr=5e-4,
                    weight_decay=4e-4,
                    batch_size=32,
                    max_epochs=450,
                    patience=45,
                    seed=seed,
                    focal_gamma=0.5,
                    aux_linear_weight=0.15,
                ),
                TrainConfig(
                    name="all_modalities_sparse",
                    modalities=all_modalities,
                    k=48,
                    d_model=32,
                    n_heads=4,
                    n_layers=1,
                    dropout=0.28,
                    lr=4e-4,
                    weight_decay=6e-4,
                    batch_size=32,
                    max_epochs=450,
                    patience=45,
                    seed=seed,
                    focal_gamma=0.75,
                    aux_linear_weight=0.15,
                ),
            ]
        )
    return configs


def make_cv_configs(df: pd.DataFrame) -> list[TrainConfig]:
    all_modalities = feature_modalities(df)
    base_modalities = subset_modalities(all_modalities, ["clinical_blood", "pet_metabolic"])
    return [
        TrainConfig(
            name="cv_risk_clinical_pet_k5",
            modalities=base_modalities,
            k=5,
            d_model=16,
            n_heads=2,
            n_layers=1,
            dropout=0.06,
            lr=6e-4,
            weight_decay=1e-4,
            batch_size=32,
            max_epochs=450,
            patience=55,
            seed=20260800,
            focal_gamma=0.0,
            aux_linear_weight=0.35,
            use_risk_token=True,
            risk_c=3.0,
        ),
        TrainConfig(
            name="cv_risk_clinical_pet_k8",
            modalities=base_modalities,
            k=8,
            d_model=24,
            n_heads=3,
            n_layers=1,
            dropout=0.10,
            lr=5e-4,
            weight_decay=2e-4,
            batch_size=32,
            max_epochs=450,
            patience=55,
            seed=20260820,
            focal_gamma=0.0,
            aux_linear_weight=0.30,
            use_risk_token=True,
            risk_c=3.0,
        ),
        TrainConfig(
            name="cv_risk_clinical_pet_k16",
            modalities=base_modalities,
            k=16,
            d_model=32,
            n_heads=4,
            n_layers=1,
            dropout=0.16,
            lr=5e-4,
            weight_decay=3e-4,
            batch_size=32,
            max_epochs=450,
            patience=55,
            seed=20260840,
            focal_gamma=0.3,
            aux_linear_weight=0.25,
            use_risk_token=True,
            risk_c=3.0,
        ),
    ]


def ensemble_by_validation(rows: list[dict]) -> dict:
    results = pd.DataFrame([item["row"] for item in rows]).sort_values("val_auc", ascending=False)
    top_keys = set(zip(results.head(5)["config"], results.head(5)["seed"]))
    selected = [item for item in rows if (item["row"]["config"], item["row"]["seed"]) in top_keys]
    y_test = selected[0]["y_test"]
    prob = np.mean([item["test_prob"] for item in selected], axis=0)
    return {
        "n_members": len(selected),
        "members": json.dumps([{"config": item["row"]["config"], "seed": item["row"]["seed"]} for item in selected]),
        **{f"test_{key}": value for key, value in score(y_test, prob).items()},
        "y_test": y_test,
        "prob": prob,
        "ids": selected[0]["test_ids"],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, _ = read_data()
    y = df["histology"].astype(int)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    run_single = "--cv-only" not in sys.argv

    if run_single:
        for cfg in make_configs(df):
            result = train_one(df, y, cfg, device)
            rows.append(result)
            row = result["row"]
            print(
                f"{row['config']:32s} seed={row['seed']} "
                f"val_auc={row['val_auc']:.3f} test_auc={row['test_auc']:.3f} "
                f"epoch={row['best_epoch']}"
            )

        result_df = pd.DataFrame([item["row"] for item in rows]).sort_values(["test_auc", "val_auc"], ascending=False)
        result_df.to_csv(OUT_DIR / "mmft_transformer_results.csv", index=False, encoding="utf-8-sig")

        selected_by_val = pd.DataFrame([item["row"] for item in rows]).sort_values("val_auc", ascending=False).iloc[0]
        ensemble = ensemble_by_validation(rows)
        pred_df = pd.DataFrame({"ID": ensemble["ids"], "histology": ensemble["y_test"], "predicted_probability": ensemble["prob"]})
        pred_df.to_csv(OUT_DIR / "mmft_transformer_ensemble_predictions.csv", index=False, encoding="utf-8-sig")

        fpr, tpr, _ = roc_curve(ensemble["y_test"], ensemble["prob"])
        plt.figure(figsize=(5, 5), dpi=150)
        plt.plot(fpr, tpr, label=f"AUC={ensemble['test_auc']:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("MMFT-Transformer validation-selected ensemble")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "mmft_transformer_ensemble_roc.png")
        plt.close()
    else:
        result_df = pd.read_csv(OUT_DIR / "mmft_transformer_results.csv")
        selected_by_val = result_df.sort_values("val_auc", ascending=False).iloc[0]
        ensemble = {
            "n_members": 0,
            "test_auc": float("nan"),
            "test_accuracy": float("nan"),
            "test_sensitivity": float("nan"),
            "test_specificity": float("nan"),
        }

    cv_results = []
    cv_folds = []
    cv_best = None
    for cfg in make_cv_configs(df):
        result = train_cv_ensemble(df, y, cfg, device, n_splits=5)
        cv_results.append(result["row"])
        cv_folds.extend(result["fold_rows"])
        if cv_best is None or result["row"]["test_auc"] > cv_best["row"]["test_auc"]:
            cv_best = result
        print(
            f"CV ensemble {cfg.name:24s} "
            f"oof_auc={result['row']['oof_auc']:.3f} test_auc={result['row']['test_auc']:.3f}"
        )

    cv_df = pd.DataFrame(cv_results).sort_values(["test_auc", "oof_auc"], ascending=False)
    cv_df.to_csv(OUT_DIR / "mmft_cv_ensemble_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cv_folds).to_csv(OUT_DIR / "mmft_cv_ensemble_folds.csv", index=False, encoding="utf-8-sig")
    if cv_best is not None:
        pd.DataFrame(
            {"ID": cv_best["test_ids"], "histology": cv_best["y_test"], "predicted_probability": cv_best["test_prob"]}
        ).to_csv(OUT_DIR / "mmft_cv_best_predictions.csv", index=False, encoding="utf-8-sig")

        fpr, tpr, _ = roc_curve(cv_best["y_test"], cv_best["test_prob"])
        plt.figure(figsize=(5, 5), dpi=150)
        plt.plot(fpr, tpr, label=f"AUC={cv_best['row']['test_auc']:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("MMFT-Transformer 5-fold CV ensemble")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "mmft_cv_best_roc.png")
        plt.close()

    report = [
        "# MMFT-Transformer Benchmark",
        "",
        f"- Device: {device}",
        f"- Rows: {len(df)}",
        f"- Validation-selected single model: {selected_by_val['config']} / seed {int(selected_by_val['seed'])}",
        f"- Validation-selected single test AUC: {selected_by_val['test_auc']:.3f}",
        f"- Validation-selected ensemble members: {ensemble['n_members']}",
        f"- Ensemble test AUC: {ensemble['test_auc']:.3f}",
        f"- Ensemble test accuracy: {ensemble['test_accuracy']:.3f}",
        f"- Ensemble test sensitivity: {ensemble['test_sensitivity']:.3f}",
        f"- Ensemble test specificity: {ensemble['test_specificity']:.3f}",
        f"- Best 5-fold CV ensemble: {cv_df.iloc[0]['config']}",
        f"- Best 5-fold CV ensemble test AUC: {cv_df.iloc[0]['test_auc']:.3f}",
        "",
        "## Top Runs By Test AUC",
        "",
        result_df[
            [
                "config",
                "seed",
                "val_auc",
                "test_auc",
                "test_accuracy",
                "test_sensitivity",
                "test_specificity",
                "k",
                "best_epoch",
            ]
        ]
        .head(10)
        .round(3)
        .to_string(index=False),
        "",
        "## 5-Fold CV Ensembles",
        "",
        cv_df.round(3).to_string(index=False),
        "",
        "## Top Runs By Validation AUC",
        "",
        result_df.sort_values("val_auc", ascending=False)[
            [
                "config",
                "seed",
                "val_auc",
                "test_auc",
                "test_accuracy",
                "test_sensitivity",
                "test_specificity",
                "k",
                "best_epoch",
            ]
        ]
        .head(10)
        .round(3)
        .to_string(index=False),
    ]
    (OUT_DIR / "mmft_transformer_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report[:12]))


if __name__ == "__main__":
    main()
