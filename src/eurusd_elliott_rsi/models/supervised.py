"""Supervised neural models for EUR/USD Elliott-wave and RSI features.

This module provides utilities for leakage-safe, walk-forward supervised
classification and regression experiments on daily EUR/USD data.

Supported tasks
---------------
* Classification: next-horizon direction (long/short/neutral).
* Regression: next-horizon forward return.

Supported architectures
-----------------------
* MLP over engineered tabular Elliott/RSI features.
* 1D CNN over rolling OHLC/feature windows.
* LSTM/GRU over rolling daily windows.
* Transformer encoder over rolling daily windows.

The implementation intentionally avoids random train/test shuffling.  Use
``walk_forward_splits`` or ``run_walk_forward_experiment`` to evaluate models on
chronologically ordered folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import sqrt
from typing import Any, Callable, Iterable, Literal, Sequence

import numpy as np
import pandas as pd

try:  # scikit-learn is used for scaling and metrics when available.
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover - exercised only in minimal envs.
    accuracy_score = f1_score = mean_absolute_error = mean_squared_error = None
    precision_score = recall_score = roc_auc_score = StandardScaler = None

try:  # PyTorch is optional until a model is instantiated/trained.
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - exercised only in minimal envs.
    torch = None
    nn = None
    DataLoader = TensorDataset = None

Task = Literal["classification", "regression"]
Architecture = Literal["mlp", "cnn1d", "lstm", "gru", "transformer"]


class DirectionClass(int, Enum):
    """Direction labels for classification targets."""

    SHORT = 0
    NEUTRAL = 1
    LONG = 2


@dataclass(frozen=True)
class TargetConfig:
    """Configuration for forward-looking supervised targets.

    Attributes:
        horizon: Number of daily bars to look ahead.
        neutral_threshold: Absolute forward-return threshold below which a move
            is labelled neutral. For example, ``0.0005`` is roughly five pips
            for EUR/USD when prices are near 1.0.
        close_col: Column containing close prices.
    """

    horizon: int = 5
    neutral_threshold: float = 0.0
    close_col: str = "close"


@dataclass(frozen=True)
class WalkForwardConfig:
    """Chronological split configuration.

    ``train_size``, ``test_size``, and ``step_size`` are expressed in rows after
    target/window construction. ``expanding=True`` grows the training span from
    the first row; otherwise, a fixed-size rolling training span is used.
    """

    train_size: int
    test_size: int
    step_size: int | None = None
    expanding: bool = True
    min_train_size: int | None = None


@dataclass
class TrainingConfig:
    """Neural-network training hyperparameters."""

    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    patience: int = 8
    validation_fraction: float = 0.15
    seed: int = 42
    device: str | None = None
    extra_model_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class FoldResult:
    """Metrics and predictions for one walk-forward fold."""

    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    metrics: dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray | None = None


def _require_torch() -> None:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise ImportError("PyTorch is required for neural supervised models.")


def _require_sklearn() -> None:
    if StandardScaler is None:
        raise ImportError("scikit-learn is required for scaling and metrics.")


def make_forward_return(df: pd.DataFrame, config: TargetConfig = TargetConfig()) -> pd.Series:
    """Return ``close[t+horizon] / close[t] - 1`` indexed like ``df``."""

    if config.horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if config.close_col not in df:
        raise KeyError(f"missing close column: {config.close_col!r}")
    close = df[config.close_col].astype(float)
    return close.shift(-config.horizon).div(close).sub(1.0).rename("forward_return")


def make_direction_target(df: pd.DataFrame, config: TargetConfig = TargetConfig()) -> pd.Series:
    """Create long/short/neutral labels from the next-N-day forward return.

    Labels are encoded as ``SHORT=0``, ``NEUTRAL=1``, and ``LONG=2``.
    """

    fwd = make_forward_return(df, config)
    labels = np.full(len(fwd), DirectionClass.NEUTRAL.value, dtype=float)
    labels[fwd.to_numpy() > config.neutral_threshold] = DirectionClass.LONG.value
    labels[fwd.to_numpy() < -config.neutral_threshold] = DirectionClass.SHORT.value
    return pd.Series(labels, index=df.index, name="direction")


def walk_forward_splits(n_samples: int, config: WalkForwardConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build leakage-safe chronological train/test index splits."""

    if config.train_size <= 0 or config.test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    step = config.step_size or config.test_size
    if step <= 0:
        raise ValueError("step_size must be positive")

    min_train = config.min_train_size or config.train_size
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    fold_start = config.train_size
    while fold_start + config.test_size <= n_samples:
        train_start = 0 if config.expanding else fold_start - config.train_size
        train_end = fold_start
        if train_end - train_start >= min_train:
            test_start = fold_start
            test_end = fold_start + config.test_size
            splits.append((np.arange(train_start, train_end), np.arange(test_start, test_end)))
        fold_start += step
    if not splits:
        raise ValueError("no walk-forward splits could be created; reduce train/test sizes")
    return splits


def build_tabular_dataset(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    task: Task,
    target_config: TargetConfig = TargetConfig(),
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Build a tabular feature matrix for the MLP architecture."""

    target = make_direction_target(df, target_config) if task == "classification" else make_forward_return(df, target_config)
    data = pd.concat([df.loc[:, feature_cols], target], axis=1).dropna()
    X = data.loc[:, feature_cols].astype(float).to_numpy()
    y = data.iloc[:, -1].to_numpy(dtype=np.int64 if task == "classification" else float)
    return X, y, data.index


def build_sequence_dataset(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    task: Task,
    window: int,
    target_config: TargetConfig = TargetConfig(),
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Build rolling-window sequence samples for CNN/RNN/Transformer models.

    Each sample ending at time ``t`` uses rows ``t-window+1`` through ``t`` and
    predicts the target generated from ``close[t+horizon]``.
    """

    if window <= 1:
        raise ValueError("window must be greater than one for sequence models")
    target = make_direction_target(df, target_config) if task == "classification" else make_forward_return(df, target_config)
    values = df.loc[:, feature_cols].astype(float).to_numpy()
    targets = target.to_numpy()
    valid_feature_rows = np.isfinite(values).all(axis=1)
    X, y, idx = [], [], []
    for end in range(window - 1, len(df)):
        start = end - window + 1
        if not valid_feature_rows[start : end + 1].all() or not np.isfinite(targets[end]):
            continue
        X.append(values[start : end + 1])
        y.append(targets[end])
        idx.append(df.index[end])
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64 if task == "classification" else np.float32),
        pd.Index(idx),
    )


class MLP(nn.Module):
    """Multilayer perceptron for engineered Elliott/RSI feature vectors."""

    def __init__(self, n_features: int, output_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for _ in range(max(1, num_layers)):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class CNN1D(nn.Module):
    """1D CNN over rolling OHLC/Elliott/RSI feature windows."""

    def __init__(self, n_features: int, output_dim: int, hidden_dim: int = 64, dropout: float = 0.1, kernel_size: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x.transpose(1, 2))


class RecurrentModel(nn.Module):
    """LSTM/GRU sequence model over rolling daily windows."""

    def __init__(self, cell: Literal["lstm", "gru"], n_features: int, output_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class TransformerSequenceModel(nn.Module):
    """Transformer encoder for sequence classification or regression."""

    def __init__(self, n_features: int, output_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1, nhead: int = 4):
        super().__init__()
        if hidden_dim % nhead != 0:
            raise ValueError("hidden_dim must be divisible by nhead")
        self.input_projection = nn.Linear(n_features, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        encoded = self.encoder(self.input_projection(x))
        return self.head(encoded[:, -1, :])


def make_model(architecture: Architecture, n_features: int, task: Task, config: TrainingConfig) -> "nn.Module":
    """Instantiate one of the supported neural architectures."""

    _require_torch()
    output_dim = 3 if task == "classification" else 1
    kwargs = dict(hidden_dim=config.hidden_dim, num_layers=config.num_layers, dropout=config.dropout)
    kwargs.update(config.extra_model_kwargs)
    if architecture == "mlp":
        return MLP(n_features, output_dim, **kwargs)
    if architecture == "cnn1d":
        kwargs.pop("num_layers", None)
        return CNN1D(n_features, output_dim, **kwargs)
    if architecture in {"lstm", "gru"}:
        return RecurrentModel(architecture, n_features, output_dim, **kwargs)
    if architecture == "transformer":
        return TransformerSequenceModel(n_features, output_dim, **kwargs)
    raise ValueError(f"unknown architecture: {architecture}")


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict[str, float]:
    """Accuracy, precision, recall, F1, and ROC-AUC where computable."""

    _require_sklearn()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, y_score, multi_class="ovr", labels=[0, 1, 2]))
        except ValueError:
            pass
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, directional accuracy, and Pearson correlation."""

    _require_sklearn()
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    true = np.asarray(y_true, dtype=float).reshape(-1)
    corr = np.corrcoef(true, pred)[0, 1] if len(true) > 1 and np.std(true) > 0 and np.std(pred) > 0 else np.nan
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(sqrt(mean_squared_error(true, pred))),
        "directional_accuracy": float(np.mean(np.sign(true) == np.sign(pred))),
        "correlation": float(corr),
    }


def _scale_fold(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _require_sklearn()
    scaler = StandardScaler()
    if X_train.ndim == 2:
        return scaler.fit_transform(X_train), scaler.transform(X_test)
    n_train, window, n_features = X_train.shape
    train_2d = X_train.reshape(-1, n_features)
    scaler.fit(train_2d)
    return (
        scaler.transform(train_2d).reshape(n_train, window, n_features),
        scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape[0], window, n_features),
    )


def _train_predict_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    task: Task,
    architecture: Architecture,
    config: TrainingConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    _require_torch()
    torch.manual_seed(config.seed)
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    n_features = X_train.shape[-1] if X_train.ndim == 3 else X_train.shape[1]
    model = make_model(architecture, n_features, task, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()

    val_size = int(len(X_train) * config.validation_fraction)
    val_size = val_size if val_size >= 1 and len(X_train) - val_size >= 1 else 0
    if val_size:
        X_fit, X_val = X_train[:-val_size], X_train[-val_size:]
        y_fit, y_val = y_train[:-val_size], y_train[-val_size:]
    else:
        X_fit, X_val, y_fit, y_val = X_train, None, y_train, None

    y_dtype = torch.long if task == "classification" else torch.float32
    dataset = TensorDataset(torch.as_tensor(X_fit, dtype=torch.float32), torch.as_tensor(y_fit, dtype=y_dtype))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    best_state, best_loss, stale = None, float("inf"), 0

    for _ in range(config.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb) if task == "classification" else criterion(out.squeeze(-1), yb)
            loss.backward()
            optimizer.step()
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
                yv = torch.as_tensor(y_val, dtype=y_dtype, device=device)
                out = model(xv)
                val_loss = criterion(out, yv) if task == "classification" else criterion(out.squeeze(-1), yv)
                score = float(val_loss.detach().cpu())
            if score < best_loss:
                best_loss, stale = score, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= config.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X_test, dtype=torch.float32, device=device)).detach().cpu()
    if task == "classification":
        probs = torch.softmax(logits, dim=1).numpy()
        return probs.argmax(axis=1), probs
    return logits.squeeze(-1).numpy(), None


def run_walk_forward_experiment(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    task: Task,
    architecture: Architecture,
    target_config: TargetConfig = TargetConfig(),
    split_config: WalkForwardConfig | None = None,
    training_config: TrainingConfig = TrainingConfig(),
    window: int = 30,
) -> list[FoldResult]:
    """Train/evaluate a neural model over walk-forward splits.

    Use ``architecture='mlp'`` for engineered tabular features.  Use
    ``'cnn1d'``, ``'lstm'``, ``'gru'``, or ``'transformer'`` for rolling windows.
    """

    if architecture == "mlp":
        X, y, _ = build_tabular_dataset(df, feature_cols, task, target_config)
    else:
        X, y, _ = build_sequence_dataset(df, feature_cols, task, window, target_config)
    split_config = split_config or WalkForwardConfig(train_size=max(100, len(y) // 3), test_size=max(20, len(y) // 10))
    results: list[FoldResult] = []
    for fold, (train_idx, test_idx) in enumerate(walk_forward_splits(len(y), split_config), start=1):
        X_train, X_test = _scale_fold(X[train_idx], X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]
        y_pred, y_score = _train_predict_fold(X_train, y_train, X_test, task, architecture, training_config)
        metrics = classification_metrics(y_test, y_pred, y_score) if task == "classification" else regression_metrics(y_test, y_pred)
        results.append(FoldResult(fold, int(train_idx[0]), int(train_idx[-1] + 1), int(test_idx[0]), int(test_idx[-1] + 1), metrics, y_test, y_pred, y_score))
    return results


def summarize_folds(results: Iterable[FoldResult]) -> pd.DataFrame:
    """Return one DataFrame row per fold plus a mean metric row."""

    rows = [{"fold": r.fold, **r.metrics} for r in results]
    summary = pd.DataFrame(rows)
    if not summary.empty:
        mean_row = {"fold": "mean", **summary.drop(columns=["fold"]).mean(numeric_only=True).to_dict()}
        summary = pd.concat([summary, pd.DataFrame([mean_row])], ignore_index=True)
    return summary
