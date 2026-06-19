"""Walk-forward unsupervised market-regime detection for EUR/USD RSI systems.

The utilities in this module intentionally fit every unsupervised model only on
historical observations available at each rebalance date.  That makes the
resulting regime labels safe to use as conditioning inputs for a baseline RSI or
Elliott-wave-inspired strategy without leaking future market structure into the
backtest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module, util
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

ClusterMethod = Literal["kmeans", "gmm", "autoencoder_kmeans", "hdbscan"]
ScalingMethod = Literal["standard", "robust"]


@dataclass(frozen=True)
class RegimeFeatureConfig:
    """Configuration for engineered RSI, trend, volatility and wave features."""

    price_col: str = "close"
    high_col: str = "high"
    low_col: str = "low"
    rsi_period: int = 14
    volatility_windows: tuple[int, ...] = (10, 20, 60)
    return_windows: tuple[int, ...] = (1, 5, 20)
    trend_windows: tuple[int, ...] = (10, 20, 50)
    wave_windows: tuple[int, ...] = (5, 13, 34)
    min_periods: int = 20


@dataclass(frozen=True)
class WalkForwardRegimeConfig:
    """Walk-forward fitting schedule and clustering choices."""

    method: ClusterMethod = "kmeans"
    n_regimes: int = 4
    train_window: int = 1_500
    refit_every: int = 20
    min_train_size: int = 300
    scaling: ScalingMethod = "robust"
    random_state: int = 42
    autoencoder_latent_dim: int = 3
    autoencoder_epochs: int = 40
    hdbscan_min_cluster_size: int = 50
    favorable_quantile: float = 0.5
    min_regime_observations: int = 30


@dataclass
class RegimeModelSnapshot:
    """A model fitted on one historical walk-forward window."""

    fitted_at: pd.Timestamp | int
    method: ClusterMethod
    feature_columns: list[str]
    center_: pd.Series
    scale_: pd.Series
    estimator: Any
    encoder: Any | None = None
    training_regime_stats: pd.DataFrame | None = None


@dataclass
class WalkForwardRegimeResult:
    """Regime labels, fitted snapshots and baseline conditioning metadata."""

    regimes: pd.Series
    features: pd.DataFrame
    snapshots: list[RegimeModelSnapshot] = field(default_factory=list)
    favorable_regimes: set[int] = field(default_factory=set)
    regime_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    conditioned_strategy: pd.DataFrame = field(default_factory=pd.DataFrame)


def engineer_regime_features(
    data: pd.DataFrame,
    config: RegimeFeatureConfig | None = None,
) -> pd.DataFrame:
    """Build leakage-free features for unsupervised regime detection.

    Parameters
    ----------
    data:
        OHLC or close-only price frame.  A ``close`` column is required by
        default; ``high``/``low`` enrich Elliott-wave range features when
        present.
    config:
        Feature engineering parameters.
    """

    cfg = config or RegimeFeatureConfig()
    if cfg.price_col not in data:
        raise KeyError(f"Missing required price column: {cfg.price_col!r}")

    close = data[cfg.price_col].astype(float)
    high = data[cfg.high_col].astype(float) if cfg.high_col in data else close
    low = data[cfg.low_col].astype(float) if cfg.low_col in data else close
    returns = close.pct_change()

    features = pd.DataFrame(index=data.index)
    rsi = _rsi(close, cfg.rsi_period)
    features["rsi"] = rsi
    features["rsi_centered"] = (rsi - 50.0) / 50.0
    features["rsi_velocity"] = rsi.diff()
    features["rsi_reversal_pressure"] = np.where(rsi > 70, rsi - 70, np.where(rsi < 30, rsi - 30, 0.0))

    for window in cfg.return_windows:
        features[f"return_{window}"] = close.pct_change(window)
        features[f"return_z_{window}"] = returns / returns.rolling(window, min_periods=max(2, window // 2)).std()

    for window in cfg.volatility_windows:
        vol = returns.rolling(window, min_periods=max(2, window // 2)).std()
        features[f"volatility_{window}"] = vol
        features[f"parkinson_vol_{window}"] = np.sqrt((np.log(high / low) ** 2).rolling(window).mean() / (4.0 * np.log(2.0)))
        features[f"volatility_ratio_{window}"] = vol / vol.rolling(window * 3, min_periods=window).median()

    for window in cfg.trend_windows:
        ma = close.rolling(window, min_periods=max(2, window // 2)).mean()
        features[f"trend_distance_{window}"] = close / ma - 1.0
        features[f"trend_slope_{window}"] = ma.pct_change(window)
        features[f"trend_efficiency_{window}"] = close.diff(window).abs() / close.diff().abs().rolling(window).sum()

    for window in cfg.wave_windows:
        roll_high = high.rolling(window, min_periods=max(2, window // 2)).max()
        roll_low = low.rolling(window, min_periods=max(2, window // 2)).min()
        wave_range = (roll_high - roll_low).replace(0.0, np.nan)
        features[f"wave_position_{window}"] = (close - roll_low) / wave_range
        features[f"wave_amplitude_{window}"] = wave_range / close
        features[f"wave_impulse_{window}"] = close.diff(window) / wave_range
        features[f"wave_retrace_{window}"] = close.diff(max(1, window // 2)) / close.diff(window).replace(0.0, np.nan)

    features = features.replace([np.inf, -np.inf], np.nan)
    return features.dropna(thresh=cfg.min_periods).ffill().dropna()


class WalkForwardRegimeDetector:
    """Fit unsupervised regimes with expanding/walk-forward historical windows."""

    def __init__(self, config: WalkForwardRegimeConfig | None = None) -> None:
        self.config = config or WalkForwardRegimeConfig()

    def fit_predict(
        self,
        features: pd.DataFrame,
        forward_returns: pd.Series | None = None,
        baseline_signals: pd.Series | None = None,
    ) -> WalkForwardRegimeResult:
        """Assign regimes using only models fitted on prior data windows."""

        clean = features.replace([np.inf, -np.inf], np.nan).dropna()
        regimes = pd.Series(np.nan, index=clean.index, name="regime", dtype="float64")
        snapshots: list[RegimeModelSnapshot] = []

        for start in range(self.config.min_train_size, len(clean), self.config.refit_every):
            end = min(start + self.config.refit_every, len(clean))
            train_start = max(0, start - self.config.train_window)
            train = clean.iloc[train_start:start]
            test = clean.iloc[start:end]
            if train.empty or test.empty:
                continue
            snapshot = self._fit_snapshot(train, forward_returns)
            regimes.loc[test.index] = self._predict_snapshot(snapshot, test)
            snapshots.append(snapshot)

        regimes = regimes.astype("Int64")
        stats = summarize_regimes(regimes, forward_returns, baseline_signals)
        favorable = discover_favorable_regimes(stats, self.config.favorable_quantile, self.config.min_regime_observations)
        conditioned = condition_baseline_strategy(baseline_signals, regimes, stats, favorable) if baseline_signals is not None else pd.DataFrame(index=clean.index)
        return WalkForwardRegimeResult(regimes, clean, snapshots, favorable, stats, conditioned)

    def _fit_snapshot(self, train: pd.DataFrame, forward_returns: pd.Series | None) -> RegimeModelSnapshot:
        scaled, center, scale = _scale(train, self.config.scaling)
        estimator: Any
        encoder: Any | None = None

        if self.config.method == "kmeans":
            cluster = _sklearn_cluster()
            estimator = cluster.KMeans(n_clusters=self.config.n_regimes, n_init=20, random_state=self.config.random_state).fit(scaled)
        elif self.config.method == "gmm":
            mixture = _sklearn_mixture()
            estimator = mixture.GaussianMixture(n_components=self.config.n_regimes, covariance_type="full", random_state=self.config.random_state).fit(scaled)
        elif self.config.method == "autoencoder_kmeans":
            latent, encoder = _fit_autoencoder_embedding(scaled, self.config)
            cluster = _sklearn_cluster()
            estimator = cluster.KMeans(n_clusters=self.config.n_regimes, n_init=20, random_state=self.config.random_state).fit(latent)
        elif self.config.method == "hdbscan":
            hdbscan = _optional_module("hdbscan")
            estimator = hdbscan.HDBSCAN(min_cluster_size=self.config.hdbscan_min_cluster_size, prediction_data=True).fit(scaled)
        else:
            raise ValueError(f"Unsupported clustering method: {self.config.method}")

        snapshot = RegimeModelSnapshot(train.index[-1], self.config.method, list(train.columns), center, scale, estimator, encoder)
        train_labels = pd.Series(self._predict_snapshot(snapshot, train), index=train.index, name="regime")
        snapshot.training_regime_stats = summarize_regimes(train_labels, forward_returns)
        return snapshot

    def _predict_snapshot(self, snapshot: RegimeModelSnapshot, data: pd.DataFrame) -> np.ndarray:
        scaled = ((data[snapshot.feature_columns] - snapshot.center_) / snapshot.scale_).fillna(0.0).to_numpy()
        if snapshot.method == "gmm":
            return snapshot.estimator.predict(scaled)
        if snapshot.method == "autoencoder_kmeans":
            latent = snapshot.encoder.predict(scaled, verbose=0)
            return snapshot.estimator.predict(latent)
        if snapshot.method == "hdbscan":
            prediction = _optional_module("hdbscan.prediction")
            labels, _ = prediction.approximate_predict(snapshot.estimator, scaled)
            return labels
        return snapshot.estimator.predict(scaled)


def condition_baseline_strategy(
    baseline_signals: pd.Series | Iterable[float],
    regimes: pd.Series,
    regime_stats: pd.DataFrame,
    favorable_regimes: set[int] | None = None,
    base_oversold: float = 30.0,
    base_overbought: float = 70.0,
) -> pd.DataFrame:
    """Gate signals, adapt RSI thresholds and size positions by regime risk."""

    if isinstance(baseline_signals, pd.Series):
        signals = baseline_signals.rename("baseline_signal").reindex(regimes.index).fillna(0.0)
    else:
        signals = pd.Series(baseline_signals, index=regimes.index, name="baseline_signal").fillna(0.0)
    favorable = favorable_regimes or set(regime_stats.index.astype(int)) if not regime_stats.empty else set()
    conditioned = pd.DataFrame(index=regimes.index)
    conditioned["regime"] = regimes
    conditioned["baseline_signal"] = signals
    conditioned["trade_enabled"] = regimes.isin(favorable).fillna(False)
    conditioned["conditioned_signal"] = signals.where(conditioned["trade_enabled"], 0.0)

    volatility = regime_stats.get("forward_volatility", pd.Series(dtype=float)).replace(0.0, np.nan)
    median_vol = volatility.median() if not volatility.empty else np.nan
    size_map = (median_vol / volatility).clip(0.25, 2.0).to_dict() if pd.notna(median_vol) else {}
    edge_map = regime_stats.get("mean_forward_return", pd.Series(dtype=float)).to_dict()
    conditioned["position_size_multiplier"] = regimes.map(size_map).fillna(1.0)
    conditioned["conditioned_position"] = conditioned["conditioned_signal"] * conditioned["position_size_multiplier"]
    conditioned["regime_edge"] = regimes.map(edge_map)
    conditioned["rsi_oversold"] = base_oversold - conditioned["regime_edge"].clip(-0.001, 0.001).fillna(0.0) * 5_000
    conditioned["rsi_overbought"] = base_overbought - conditioned["regime_edge"].clip(-0.001, 0.001).fillna(0.0) * 5_000
    return conditioned


def summarize_regimes(
    regimes: pd.Series,
    forward_returns: pd.Series | None,
    baseline_signals: pd.Series | None = None,
) -> pd.DataFrame:
    """Summarize realized regime quality from already-known historical labels."""

    frame = pd.DataFrame({"regime": regimes}).dropna()
    if forward_returns is not None:
        frame["forward_return"] = forward_returns.reindex(frame.index)
    if baseline_signals is not None and forward_returns is not None:
        frame["strategy_return"] = baseline_signals.reindex(frame.index).fillna(0.0) * frame["forward_return"]
    if frame.empty:
        return pd.DataFrame()

    grouped = frame.groupby("regime")
    stats = grouped.size().to_frame("observations")
    if "forward_return" in frame:
        stats["mean_forward_return"] = grouped["forward_return"].mean()
        stats["forward_volatility"] = grouped["forward_return"].std()
        stats["sharpe_proxy"] = stats["mean_forward_return"] / stats["forward_volatility"].replace(0.0, np.nan)
    if "strategy_return" in frame:
        stats["mean_strategy_return"] = grouped["strategy_return"].mean()
        stats["strategy_hit_rate"] = grouped["strategy_return"].apply(lambda x: (x > 0).mean())
    return stats.sort_index()


def discover_favorable_regimes(stats: pd.DataFrame, quantile: float = 0.5, min_observations: int = 30) -> set[int]:
    """Select regimes whose historical edge is above the requested quantile."""

    if stats.empty:
        return set()
    score_col = "mean_strategy_return" if "mean_strategy_return" in stats else "mean_forward_return"
    if score_col not in stats:
        return set(stats.index.astype(int))
    eligible = stats[stats["observations"] >= min_observations]
    if eligible.empty:
        return set()
    cutoff = eligible[score_col].quantile(quantile)
    return set(eligible[eligible[score_col] >= cutoff].index.astype(int))


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0).ewm(alpha=1 / period, adjust=False).mean()
    losses = (-delta.clip(upper=0.0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gains / losses.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _scale(data: pd.DataFrame, method: ScalingMethod) -> tuple[np.ndarray, pd.Series, pd.Series]:
    if method == "robust":
        center = data.median()
        scale = (data.quantile(0.75) - data.quantile(0.25)).replace(0.0, np.nan)
    else:
        center = data.mean()
        scale = data.std().replace(0.0, np.nan)
    scale = scale.fillna(1.0)
    return ((data - center) / scale).fillna(0.0).to_numpy(), center, scale


def _sklearn_cluster() -> Any:
    if util.find_spec("sklearn") is None or util.find_spec("sklearn.cluster") is None:
        raise ImportError("scikit-learn is required for k-means clustering")
    return import_module("sklearn.cluster")


def _sklearn_mixture() -> Any:
    if util.find_spec("sklearn") is None or util.find_spec("sklearn.mixture") is None:
        raise ImportError("scikit-learn is required for Gaussian mixture models")
    return import_module("sklearn.mixture")


def _optional_module(name: str) -> Any:
    parent = name.split(".", 1)[0]
    if util.find_spec(parent) is None or util.find_spec(name) is None:
        raise ImportError(f"Optional dependency {name!r} is not installed")
    return import_module(name)


def _fit_autoencoder_embedding(scaled: np.ndarray, config: WalkForwardRegimeConfig) -> tuple[np.ndarray, Any]:
    keras = _optional_module("tensorflow.keras")
    layers = keras.layers
    models = keras.models
    callbacks = keras.callbacks
    input_dim = scaled.shape[1]
    latent_dim = min(config.autoencoder_latent_dim, max(1, input_dim // 2))
    inputs = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(max(latent_dim * 2, 4), activation="relu")(inputs)
    latent = layers.Dense(latent_dim, activation="linear", name="latent_regime_embedding")(encoded)
    decoded = layers.Dense(max(latent_dim * 2, 4), activation="relu")(latent)
    outputs = layers.Dense(input_dim, activation="linear")(decoded)
    autoencoder = models.Model(inputs, outputs)
    encoder = models.Model(inputs, latent)
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(
        scaled,
        scaled,
        epochs=config.autoencoder_epochs,
        batch_size=min(128, max(16, len(scaled) // 10)),
        verbose=0,
        callbacks=[callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)],
    )
    return encoder.predict(scaled, verbose=0), encoder
