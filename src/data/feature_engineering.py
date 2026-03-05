"""
Feature engineering transformations for GPON router TR-069 telemetry data.

All public functions accept a pandas DataFrame that has already been
validated and is sorted by Device_ID and Timestamp (ascending).
They return a new DataFrame with additional derived columns.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

MILLIVOLTS_TO_VOLTS: float = 1_000.0
WINDOW_24H: int = 24
WINDOW_12H: int = 12
WINDOW_48H: int = 48


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Required columns missing from DataFrame: {missing}. Available: {df.columns.tolist()}")


def normalize_voltage(df: pd.DataFrame) -> pd.DataFrame:
    """Add Voltage_V column by converting Voltage_mV to Volts."""
    _require_columns(df, ["Voltage_mV"])
    result = df.copy()
    result["Voltage_V"] = result["Voltage_mV"] / MILLIVOLTS_TO_VOLTS
    logger.debug("normalize_voltage: added Voltage_V column.")
    return result


def add_rolling_optical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-device rolling statistics on optical power signals.

    Adds: RX_Power_24h_Mean, RX_Power_24h_Std, TX_Power_24h_Mean,
    TX_Power_24h_Std, RX_TX_Power_Delta, RX_Power_48h_Mean.
    """
    _require_columns(df, ["Device_ID", "Optical_RX_Power_dBm", "Optical_TX_Power_dBm"])
    result = df.copy()
    grouped_rx = result.groupby("Device_ID", sort=False)["Optical_RX_Power_dBm"]
    grouped_tx = result.groupby("Device_ID", sort=False)["Optical_TX_Power_dBm"]

    result["RX_Power_24h_Mean"] = grouped_rx.transform(lambda s: s.rolling(WINDOW_24H, min_periods=1).mean())
    result["RX_Power_24h_Std"] = grouped_rx.transform(lambda s: s.rolling(WINDOW_24H, min_periods=2).std().fillna(0.0))
    result["TX_Power_24h_Mean"] = grouped_tx.transform(lambda s: s.rolling(WINDOW_24H, min_periods=1).mean())
    result["TX_Power_24h_Std"] = grouped_tx.transform(lambda s: s.rolling(WINDOW_24H, min_periods=2).std().fillna(0.0))
    result["RX_TX_Power_Delta"] = result["Optical_RX_Power_dBm"] - result["Optical_TX_Power_dBm"]
    result["RX_Power_48h_Mean"] = grouped_rx.transform(lambda s: s.rolling(WINDOW_48H, min_periods=1).mean())

    logger.debug("add_rolling_optical_features: added 6 optical feature columns.")
    return result


def add_temperature_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a linear OLS slope of temperature over a 12-hour rolling window.

    Adds: Temp_Trend_Slope_12h.
    """
    _require_columns(df, ["Device_ID", "Temperature_C"])
    result = df.copy()

    def _rolling_slope(series: pd.Series) -> pd.Series:
        slopes = np.full(len(series), np.nan)
        arr = series.to_numpy(dtype=float)
        x = np.arange(WINDOW_12H, dtype=float)
        for i in range(WINDOW_12H - 1, len(arr)):
            window = arr[i - WINDOW_12H + 1 : i + 1]
            if np.isnan(window).any():
                continue
            slope, *_ = stats.linregress(x, window)
            slopes[i] = slope
        if len(arr) >= 2:
            slopes[: WINDOW_12H - 1] = np.gradient(arr[: WINDOW_12H - 1])
        return pd.Series(slopes, index=series.index)

    result["Temp_Trend_Slope_12h"] = result.groupby("Device_ID", sort=False)["Temperature_C"].transform(_rolling_slope)
    logger.debug("add_temperature_trend: added Temp_Trend_Slope_12h column.")
    return result


def add_error_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive interface-error rate and reboot instability features.

    Adds: Error_Rate_24h, Reboot_Instability.
    """
    _require_columns(df, ["Device_ID", "Interface_Error_Count", "Reboot_Count_Last_7D", "Maintenance_Count_Last_30D"])
    result = df.copy()
    result["Error_Rate_24h"] = result.groupby("Device_ID", sort=False)["Interface_Error_Count"].transform(
        lambda s: s.rolling(WINDOW_24H, min_periods=1).sum()
    )
    result["Reboot_Instability"] = result["Reboot_Count_Last_7D"] * result["Maintenance_Count_Last_30D"]
    logger.debug("add_error_rate_features: added Error_Rate_24h and Reboot_Instability.")
    return result


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature-engineering transformations in the correct order.

    Args:
        df: Validated telemetry DataFrame with Device_ID and Timestamp columns.

    Returns:
        Fully engineered feature DataFrame.
    """
    _require_columns(df, ["Device_ID", "Timestamp"])
    df_sorted = df.sort_values(["Device_ID", "Timestamp"], kind="stable").reset_index(drop=True)
    logger.info("build_feature_matrix: starting with %d rows x %d cols.", *df_sorted.shape)

    pipeline = [normalize_voltage, add_rolling_optical_features, add_temperature_trend, add_error_rate_features]
    result = df_sorted
    for transform in pipeline:
        result = transform(result)
        logger.debug("After %s: %d cols.", transform.__name__, result.shape[1])

    logger.info("build_feature_matrix: finished with %d rows x %d cols (+%d new features).", result.shape[0], result.shape[1], result.shape[1] - df_sorted.shape[1])
    return result
