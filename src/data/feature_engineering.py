"""
Feature engineering transformations for GPON router TR-069 telemetry data.

All public functions accept a :class:`pandas.DataFrame` that has already been
validated and is **sorted by ``Device_ID`` and ``Timestamp``** (ascending).
They return a new DataFrame with additional derived columns; the original
columns are never mutated.

Rolling-window features use ``groupby(...).transform(...)`` so that per-device
history never bleeds across device boundaries.

Typical call order
------------------
1. :func:`normalize_voltage`
2. :func:`add_rolling_optical_features`
3. :func:`add_temperature_trend`
4. :func:`add_error_rate_features`
5. :func:`build_feature_matrix` — orchestrates all of the above
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats  # scipy is a transitive dep of scikit-learn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MILLIVOLTS_TO_VOLTS: float = 1_000.0

# Rolling window sizes expressed as number of rows.  Adjust if sampling rate
# changes (assumed: one reading per hour).
WINDOW_24H: int = 24   # 24 rows  ≈ 24 hours at 1-sample/hour cadence
WINDOW_12H: int = 12   # 12 rows  ≈ 12 hours
WINDOW_48H: int = 48   # 48 rows  ≈ 48 hours


# ---------------------------------------------------------------------------
# Individual transformation functions
# ---------------------------------------------------------------------------

def normalize_voltage(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``Voltage_V`` column by converting ``Voltage_mV`` to Volts.

    Args:
        df: Input DataFrame containing a ``Voltage_mV`` column.

    Returns:
        DataFrame with an additional ``Voltage_V`` column (original unchanged).

    Raises:
        KeyError: If ``Voltage_mV`` is not present in *df*.
    """
    _require_columns(df, ["Voltage_mV"])
    result = df.copy()
    result["Voltage_V"] = result["Voltage_mV"] / MILLIVOLTS_TO_VOLTS
    logger.debug("normalize_voltage: added Voltage_V column.")
    return result

def add_rolling_optical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-device rolling statistics on optical power signals.

    New columns added
    -----------------
    - ``RX_Power_24h_Mean``  : 24-hour rolling mean of ``Optical_RX_Power_dBm``
    - ``RX_Power_24h_Std``   : 24-hour rolling std  of ``Optical_RX_Power_dBm``
    - ``TX_Power_24h_Mean``  : 24-hour rolling mean of ``Optical_TX_Power_dBm``
    - ``TX_Power_24h_Std``   : 24-hour rolling std  of ``Optical_TX_Power_dBm``
    - ``RX_TX_Power_Delta``  : Instantaneous difference RX − TX (link loss proxy)
    - ``RX_Power_48h_Mean``  : 48-hour rolling mean of ``Optical_RX_Power_dBm``

    Args:
        df: Input DataFrame sorted by ``Device_ID`` and ``Timestamp``,
            containing ``Optical_RX_Power_dBm`` and ``Optical_TX_Power_dBm``.

    Returns:
        DataFrame with additional optical rolling-feature columns.

    Raises:
        KeyError: If required source columns are absent.
    """
    required = ["Device_ID", "Optical_RX_Power_dBm", "Optical_TX_Power_dBm"]
    _require_columns(df, required)

    result = df.copy()
    grouped_rx = result.groupby("Device_ID", sort=False)["Optical_RX_Power_dBm"]
    grouped_tx = result.groupby("Device_ID", sort=False)["Optical_TX_Power_dBm"]

    result["RX_Power_24h_Mean"] = grouped_rx.transform(
        lambda s: s.rolling(WINDOW_24H, min_periods=1).mean()
    )
    result["RX_Power_24h_Std"] = grouped_rx.transform(
        lambda s: s.rolling(WINDOW_24H, min_periods=2).std().fillna(0.0)
    )
    result["TX_Power_24h_Mean"] = grouped_tx.transform(
        lambda s: s.rolling(WINDOW_24H, min_periods=1).mean()
    )
    result["TX_Power_24h_Std"] = grouped_tx.transform(
        lambda s: s.rolling(WINDOW_24H, min_periods=2).std().fillna(0.0)
    )
    result["RX_TX_Power_Delta"] = (
        result["Optical_RX_Power_dBm"] - result["Optical_TX_Power_dBm"]
    )
    result["RX_Power_48h_Mean"] = grouped_rx.transform(
        lambda s: s.rolling(WINDOW_48H, min_periods=1).mean()
    )

    logger.debug("add_rolling_optical_features: added 6 optical feature columns.")
    return result

def add_temperature_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a linear slope of temperature over a 12-hour rolling window.

    The slope is estimated via ordinary least-squares (``scipy.stats.linregress``)
    on the last *WINDOW_12H* samples per device.  A positive slope indicates a
    rising temperature trend, which is a known precursor to hardware failure.

    New columns added
    -----------------
    - ``Temp_Trend_Slope_12h``: OLS slope of ``Temperature_C`` over 12 h window.

    Args:
        df: Input DataFrame sorted by ``Device_ID`` and ``Timestamp``,
            containing ``Temperature_C``.

    Returns:
        DataFrame with the additional ``Temp_Trend_Slope_12h`` column.

    Raises:
        KeyError: If ``Temperature_C`` or ``Device_ID`` is absent.
    """
    _require_columns(df, ["Device_ID", "Temperature_C"])

    result = df.copy()

    def _rolling_slope(series: pd.Series) -> pd.Series:
        """Apply a rolling OLS slope with window size WINDOW_12H."""
        slopes = np.full(len(series), np.nan)
        arr = series.to_numpy(dtype=float)
        x = np.arange(WINDOW_12H, dtype=float)

        for i in range(WINDOW_12H - 1, len(arr)):
            window = arr[i - WINDOW_12H + 1 : i + 1]
            if np.isnan(window).any():
                continue
            slope, *_ = stats.linregress(x, window)
            slopes[i] = slope

        # Back-fill the first (WINDOW_12H − 1) positions with a simple diff
        if len(arr) >= 2:
            slopes[:WINDOW_12H - 1] = np.gradient(arr[:WINDOW_12H - 1])

        return pd.Series(slopes, index=series.index)

    result["Temp_Trend_Slope_12h"] = result.groupby(
        "Device_ID", sort=False
    )["Temperature_C"].transform(_rolling_slope)

    logger.debug("add_temperature_trend: added Temp_Trend_Slope_12h column.")
    return result

def add_error_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive interface-error rate and reboot instability features.

    New columns added
    -----------------
    - ``Error_Rate_24h``      : Rolling sum of ``Interface_Error_Count`` over 24 h.
    - ``Reboot_Instability``  : ``Reboot_Count_Last_7D`` × ``Maintenance_Count_Last_30D``
                                interaction term — amplifies devices with both frequent
                                reboots *and* high maintenance history.

    Args:
        df: Input DataFrame containing ``Interface_Error_Count``,
            ``Reboot_Count_Last_7D``, and ``Maintenance_Count_Last_30D``.

    Returns:
        DataFrame with the additional error-rate feature columns.

    Raises:
        KeyError: If required source columns are absent.
    """
    required = [
        "Device_ID",
        "Interface_Error_Count",
        "Reboot_Count_Last_7D",
        "Maintenance_Count_Last_30D",
    ]
    _require_columns(df, required)

    result = df.copy()

    result["Error_Rate_24h"] = result.groupby(
        "Device_ID", sort=False
    )["Interface_Error_Count"].transform(
        lambda s: s.rolling(WINDOW_24H, min_periods=1).sum()
    )

    result["Reboot_Instability"] = (
        result["Reboot_Count_Last_7D"] * result["Maintenance_Count_Last_30D"]
    )

    logger.debug("add_error_rate_features: added Error_Rate_24h and Reboot_Instability.")
    return result

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature-engineering transformations in the correct order.

    The function validates that the DataFrame is sorted by ``Device_ID`` and
    ``Timestamp`` and then calls each individual transform in sequence.

    Args:
        df: Raw (but Pydantic-validated) telemetry DataFrame. Must contain
            ``Device_ID`` and ``Timestamp`` columns, and all raw feature columns
            listed in :mod:`src.data.validation`.

    Returns:
        Fully engineered feature DataFrame ready for model training or serving.

    Example:
        >>> engineered_df = build_feature_matrix(validated_df)
        >>> engineered_df.shape
        (n_rows, n_original_cols + 11)
    """
    _require_columns(df, ["Device_ID", "Timestamp"])

    # Ensure correct sort order (stable sort preserves ties within a device)
    df_sorted = df.sort_values(["Device_ID", "Timestamp"], kind="stable").reset_index(
        drop=True
    )

    logger.info(
        "build_feature_matrix: starting with %d rows × %d cols.",
        df_sorted.shape[0],
        df_sorted.shape[1],
    )

    pipeline = [
        normalize_voltage,
        add_rolling_optical_features,
        add_temperature_trend,
        add_error_rate_features,
    ]

    result = df_sorted
    for transform in pipeline:
        result = transform(result)
        logger.debug(
            "After %s: %d cols.", transform.__name__, result.shape[1]
        )

    logger.info(
        "build_feature_matrix: finished with %d rows × %d cols (+%d new features).",
        result.shape[0],
        result.shape[1],
        result.shape[1] - df_sorted.shape[1],
    )
    return result

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Assert that all *columns* are present in *df*.

    Args:
        df: DataFrame to inspect.
        columns: List of column names that must exist.

    Raises:
        KeyError: With a clear message listing the missing columns.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"Required columns missing from DataFrame: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )
