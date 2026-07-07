"""Robust, name-indexed token injection for MMFT -- redo of the TCGA-guided
mechanistic-token idea (scripts/mmft_biology_guided.py), engineered this time
to make the §13 bug class (manual column-offset arithmetic, e.g.
`shape[1]-2`, silently pointing at the wrong column once more tokens are
appended) structurally impossible rather than merely hand-checked.

Design principle: NEVER compute a token's column index by arithmetic on
`tensor.shape[1]`. Instead, carry an explicit `names: list[str]` alongside
the tensor (a `NamedBatch`), append synchronised with the tensor, and always
look up a token's position via `names.index("token_name")`. A runtime
assertion cross-checks the looked-up column's actual values against the
appender's own output before it is used for residual-initialisation, so a
wiring mistake fails loudly (AssertionError) instead of silently training a
different, unintended architecture.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mmft_biology_guided import GLYCOLYSIS_FEATURES, INFLAMMATION_FEATURES, MechanisticTokenAppender
from scripts.train_mmft_transformer_petct import (
    MMFTTransformer,
    ModalityAwarePreprocessor,
    RiskTokenAppender,
    TabularBatch,
    binary_focal_bce,
    feature_modalities,
    predict,
    set_seed,
    subset_modalities,
)


@dataclass
class NamedBatch:
    batch: TabularBatch
    names: list

    def index_of(self, name: str) -> int:
        assert name in self.names, f"column '{name}' not found among {self.names}"
        idx = self.names.index(name)
        assert self.batch.x.shape[1] == len(self.names), (
            f"tensor width {self.batch.x.shape[1]} != len(names) {len(self.names)} -- "
            "names list is out of sync with the tensor, do not proceed"
        )
        return idx

    def append_column(self, col: torch.Tensor, name: str, modality_id: int) -> "NamedBatch":
        x_new = torch.cat([self.batch.x, col.view(-1, 1)], dim=1)
        modality_ids = torch.cat([
            self.batch.modality_ids,
            torch.tensor([modality_id], dtype=torch.long, device=self.batch.modality_ids.device),
        ])
        feature_ids = torch.arange(x_new.shape[1], dtype=torch.long, device=self.batch.feature_ids.device)
        new_batch = TabularBatch(x=x_new, modality_ids=modality_ids, feature_ids=feature_ids)
        return NamedBatch(batch=new_batch, names=self.names + [name])


def build_named_batches(x_tr_raw, y_tr, x_ev_raw, y_ev, k, risk_c,
                         use_risk_token=True, use_glycolysis=False, use_inflammation=False):
    """Builds train/eval NamedBatch objects with fully name-tracked columns.
    Returns (train_nb, eval_nb, pre)."""
    modalities = subset_modalities(feature_modalities(x_tr_raw), ["clinical_blood", "pet_metabolic"])
    features = [f for g in modalities.values() for f in g]
    x_tr = x_tr_raw[features].apply(pd.to_numeric, errors="coerce")
    x_ev = x_ev_raw[features].apply(pd.to_numeric, errors="coerce")

    pre = ModalityAwarePreprocessor(modalities, k=k)
    train_batch = pre.fit_transform(x_tr, y_tr)
    eval_batch = pre.transform(x_ev)
    train_nb = NamedBatch(batch=train_batch, names=list(pre.selected_features_))
    eval_nb = NamedBatch(batch=eval_batch, names=list(pre.selected_features_))

    next_modality_id = len(modalities)

    if use_risk_token:
        risk = RiskTokenAppender(modality_id=next_modality_id, c=risk_c)
        tb2 = risk.fit_transform(train_nb.batch, y_tr)
        eb2 = risk.transform(eval_nb.batch)
        # verify: the last column of tb2 must equal the appender's own computed token
        raw_logits = risk.model_.decision_function(train_nb.batch.x.detach().cpu().numpy())
        expected_token = ((raw_logits - risk.mean_) / risk.std_).astype("float32")
        actual_token = tb2.x[:, -1].detach().cpu().numpy()
        assert np.allclose(expected_token, actual_token, atol=1e-5), \
            "risk token column mismatch -- appender wiring is broken"
        train_nb = NamedBatch(batch=tb2, names=train_nb.names + ["ridge_risk_token"])
        eval_nb = NamedBatch(batch=eb2, names=eval_nb.names + ["ridge_risk_token"])
        next_modality_id += 1

    if use_glycolysis:
        glyc = MechanisticTokenAppender(GLYCOLYSIS_FEATURES, modality_id=next_modality_id, c=1.0)
        tb3 = glyc.fit_transform(x_tr_raw, y_tr, train_nb.batch)
        eb3 = glyc.transform(x_ev_raw, eval_nb.batch)
        actual_token = tb3.x[:, -1].detach().cpu().numpy()
        arr = x_tr_raw[GLYCOLYSIS_FEATURES].apply(pd.to_numeric, errors="coerce")
        arr_i = glyc.imputer_.transform(arr); arr_s = glyc.scaler_.transform(arr_i)
        expected_logits = glyc.model_.decision_function(arr_s)
        expected_token = ((expected_logits - glyc.mean_) / glyc.std_).astype("float32")
        assert np.allclose(expected_token, actual_token, atol=1e-5), \
            "glycolysis token column mismatch -- appender wiring is broken"
        train_nb = NamedBatch(batch=tb3, names=train_nb.names + ["glycolysis_token"])
        eval_nb = NamedBatch(batch=eb3, names=eval_nb.names + ["glycolysis_token"])
        next_modality_id += 1

    if use_inflammation:
        infl = MechanisticTokenAppender(INFLAMMATION_FEATURES, modality_id=next_modality_id, c=1.0)
        tb4 = infl.fit_transform(x_tr_raw, y_tr, train_nb.batch)
        eb4 = infl.transform(x_ev_raw, eval_nb.batch)
        actual_token = tb4.x[:, -1].detach().cpu().numpy()
        arr = x_tr_raw[INFLAMMATION_FEATURES].apply(pd.to_numeric, errors="coerce")
        arr_i = infl.imputer_.transform(arr); arr_s = infl.scaler_.transform(arr_i)
        expected_logits = infl.model_.decision_function(arr_s)
        expected_token = ((expected_logits - infl.mean_) / infl.std_).astype("float32")
        assert np.allclose(expected_token, actual_token, atol=1e-5), \
            "inflammation token column mismatch -- appender wiring is broken"
        train_nb = NamedBatch(batch=tb4, names=train_nb.names + ["inflammation_token"])
        eval_nb = NamedBatch(batch=eb4, names=eval_nb.names + ["inflammation_token"])
        next_modality_id += 1

    return train_nb, eval_nb, pre


def pairwise_auc_loss_local(logits, y):
    pos = logits[y == 1]; neg = logits[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return torch.tensor(0.0, device=logits.device)
    return torch.nn.functional.softplus(-(pos[:, None] - neg[None, :])).mean()


def run_named_config(x_tr_raw, y_tr, x_ev_raw, y_ev, cfg, seed, epochs):
    """cfg keys: k, d_model, n_heads, n_layers, dropout, lr, weight_decay,
    rank_weight, risk_c, focal_gamma, use_risk_token, use_glycolysis, use_inflammation."""
    set_seed(seed)
    train_nb, eval_nb, pre = build_named_batches(
        x_tr_raw, y_tr, x_ev_raw, y_ev, k=cfg["k"], risk_c=cfg["risk_c"],
        use_risk_token=cfg["use_risk_token"], use_glycolysis=cfg["use_glycolysis"],
        use_inflammation=cfg["use_inflammation"],
    )
    train_batch, eval_batch = train_nb.batch, eval_nb.batch

    model = MMFTTransformer(
        n_features=train_batch.x.shape[1],
        n_modalities=int(train_batch.modality_ids.max().item()) + 1,
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"], dropout=cfg["dropout"],
    )
    if cfg["use_risk_token"]:
        # ROBUST: look up by name, not by shape arithmetic. Verified against
        # the appender's own output inside build_named_batches() above.
        risk_idx = train_nb.index_of("ridge_risk_token")
        with torch.no_grad():
            model.linear_head.weight.zero_()
            model.linear_head.bias.zero_()
            model.linear_head.weight[0, risk_idx] = 1.0
            last_linear = None
            for module in model.deep_head.modules():
                if isinstance(module, nn.Linear):
                    last_linear = module
            if last_linear is not None:
                last_linear.weight.zero_()
                last_linear.bias.zero_()

    y_tr_np = y_tr.to_numpy().astype("float32")
    pos = y_tr_np.sum(); neg = len(y_tr_np) - pos
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    for _ in range(epochs):
        model.train()
        mini = TabularBatch(x=train_batch.x, modality_ids=train_batch.modality_ids, feature_ids=train_batch.feature_ids)
        optimizer.zero_grad(set_to_none=True)
        logits = model(mini)
        bce = binary_focal_bce(logits, torch.from_numpy(y_tr_np), pos_weight, gamma=cfg["focal_gamma"])
        rank = pairwise_auc_loss_local(logits, torch.from_numpy(y_tr_np))
        loss = (1.0 - cfg["rank_weight"]) * bce + cfg["rank_weight"] * rank
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

    from sklearn.metrics import roc_auc_score
    prob = predict(model, eval_batch)
    y_ev_np = y_ev.to_numpy().astype("float32")
    if len(np.unique(y_ev_np)) < 2:
        return None
    return float(roc_auc_score(y_ev_np, prob))
