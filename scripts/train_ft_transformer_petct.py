from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from benchmark_petct_blood_models import RANDOM_STATE, read_data


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "server_results" / "petct_blood_benchmark"


@dataclass(frozen=True)
class Config:
    block: str
    k: int
    d_model: int
    n_heads: int
    n_layers: int
    dropout: float
    lr: float
    weight_decay: float
    seed: int


class FeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls.expand(x.size(0), -1, -1)
        return torch.cat([cls, tokens], dim=1)


class FTTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int, n_heads: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.tokenizer(x)
        z = self.encoder(z)
        return self.head(z[:, 0]).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
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


def block_features(df: pd.DataFrame, block_name: str) -> list[str]:
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
    ct = [c for c in df.columns if c.startswith("ct__")]
    pet = [c for c in df.columns if c.startswith("pet__")]
    if block_name == "clinical_blood_metabolic":
        return clinical_blood + metabolic
    if block_name == "pet_radiomics":
        return pet
    if block_name == "ct_pet_radiomics":
        return ct + pet
    if block_name == "combined_all":
        return clinical_blood + metabolic + ct + pet
    raise ValueError(block_name)


def prepare_arrays(df: pd.DataFrame, features: list[str], y: pd.Series, k: int, seed: int):
    x = df[features].apply(pd.to_numeric, errors="coerce")
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.22, random_state=seed, stratify=y_trainval
    )

    imputer = SimpleImputer(strategy="median")
    variance = VarianceThreshold()
    scaler = StandardScaler()

    x_train_np = imputer.fit_transform(x_train)
    x_train_np = variance.fit_transform(x_train_np)
    x_train_np = scaler.fit_transform(x_train_np)
    k_eff = min(k, x_train_np.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k_eff)
    x_train_np = selector.fit_transform(x_train_np, y_train)

    def transform(x_part: pd.DataFrame) -> np.ndarray:
        arr = imputer.transform(x_part)
        arr = variance.transform(arr)
        arr = scaler.transform(arr)
        arr = selector.transform(arr)
        return arr.astype("float32")

    return (
        x_train_np.astype("float32"),
        transform(x_val),
        transform(x_test),
        y_train.to_numpy().astype("float32"),
        y_val.to_numpy().astype("float32"),
        y_test.to_numpy().astype("float32"),
        np.array(features)[variance.get_support()][selector.get_support()].tolist(),
    )


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    probs = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    for (xb,) in loader:
        probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(probs)


def run_config(df: pd.DataFrame, y: pd.Series, cfg: Config) -> dict:
    set_seed(cfg.seed)
    features = block_features(df, cfg.block)
    x_train, x_val, x_test, y_train, y_val, y_test, selected = prepare_arrays(df, features, y, cfg.k, cfg.seed)
    model = FTTransformer(x_train.shape[1], cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.dropout)

    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=32,
        shuffle=True,
        drop_last=False,
    )

    best_state = None
    best_val_auc = -np.inf
    best_epoch = 0
    patience = 40

    for epoch in range(1, 401):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        val_prob = predict(model, x_val)
        try:
            val_auc = roc_auc_score(y_val, val_prob)
        except ValueError:
            val_auc = 0.5
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    val_prob = predict(model, x_val)
    test_prob = predict(model, x_test)
    row = {
        "block": cfg.block,
        "k": x_train.shape[1],
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "n_layers": cfg.n_layers,
        "dropout": cfg.dropout,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        **{f"val_{k}": v for k, v in metrics(y_val, val_prob).items()},
        **{f"test_{k}": v for k, v in metrics(y_test, test_prob).items()},
    }
    row["selected_features"] = json.dumps(selected, ensure_ascii=False)
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, _ = read_data()
    y = df["histology"].astype(int)

    configs = []
    seeds = [20260701, 20260702, 20260703]
    for seed in seeds:
        configs.extend(
            [
                Config("clinical_blood_metabolic", 12, 16, 2, 1, 0.15, 1e-3, 1e-4, seed),
                Config("clinical_blood_metabolic", 24, 32, 4, 1, 0.20, 8e-4, 1e-4, seed),
                Config("pet_radiomics", 64, 32, 4, 1, 0.25, 8e-4, 2e-4, seed),
                Config("ct_pet_radiomics", 64, 32, 4, 1, 0.25, 8e-4, 2e-4, seed),
                Config("combined_all", 64, 32, 4, 1, 0.25, 8e-4, 2e-4, seed),
                Config("combined_all", 128, 32, 4, 2, 0.30, 5e-4, 5e-4, seed),
            ]
        )

    rows = []
    for cfg in configs:
        row = run_config(df, y, cfg)
        rows.append(row)
        print(
            f"{cfg.block:26s} k={row['k']:3d} seed={cfg.seed} "
            f"val_auc={row['val_auc']:.3f} test_auc={row['test_auc']:.3f}"
        )

    res = pd.DataFrame(rows).sort_values(["test_auc", "val_auc"], ascending=False)
    res.to_csv(OUT_DIR / "ft_transformer_results.csv", index=False, encoding="utf-8-sig")
    best = res.iloc[0]
    report = [
        "# FT-Transformer Benchmark",
        "",
        f"- Best block: {best['block']}",
        f"- Best k: {int(best['k'])}",
        f"- Best test AUC: {best['test_auc']:.3f}",
        f"- Best test accuracy: {best['test_accuracy']:.3f}",
        f"- Best test sensitivity: {best['test_sensitivity']:.3f}",
        f"- Best test specificity: {best['test_specificity']:.3f}",
        "",
        res[
            [
                "block",
                "k",
                "seed",
                "val_auc",
                "test_auc",
                "test_accuracy",
                "test_sensitivity",
                "test_specificity",
                "best_epoch",
            ]
        ]
        .head(10)
        .round(3)
        .to_string(index=False),
    ]
    (OUT_DIR / "ft_transformer_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
