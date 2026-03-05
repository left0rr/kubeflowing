"""
Pydantic validation models for GPON router TR-069 telemetry rows.

Each incoming telemetry record is validated against :class:`TelemetryRecord`
before entering the feature-engineering stage.  Invalid rows are logged and
dropped rather than raising hard errors so that a single bad record cannot
stall a batch ingestion job.
"""

from __future__ import annotations

import logging
from typing import Annotated

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases with field-level constraints
# ---------------------------------------------------------------------------

# Optical power range for typical SFP modules: -40 dBm … +10 dBm
OpticalPower = Annotated[float, Field(ge=-40.0, le=10.0)]

# Operating temperature range for outdoor CPE: -40 °C … +85 °C
Temperature = Annotated[float, Field(ge=-40.0, le=85.0)]

# Supply voltage range in millivolts: 2 500 mV … 4 000 mV (typical 3.3 V rail)
Voltage = Annotated[float, Field(ge=2_500.0, le=4_000.0)]

# Bias current range in milliamps: 0 mA … 100 mA
BiasCurrent = Annotated[float, Field(ge=0.0, le=100.0)]

# Non-negative integer counters
NonNegativeInt = Annotated[int, Field(ge=0)]

# Device age in days (must be positive)
DeviceAge = Annotated[int, Field(ge=1)]

# Binary classification label
BinaryLabel = Annotated[int, Field(ge=0, le=1)]


class TelemetryRecord(BaseModel):
    """Strict schema for a single GPON router TR-069 telemetry row.

    All fields are required; there are no optional fields so that downstream
    feature-engineering functions can rely on every column being present.

    Attributes:
        Device_ID: Unique device identifier string.
        Timestamp: ISO-8601 timestamp string (e.g. ``"2024-01-15T10:30:00"``).
        Optical_RX_Power_dBm: Received optical power in dBm.
        Optical_TX_Power_dBm: Transmitted optical power in dBm.
        Temperature_C: Device temperature in degrees Celsius.
        Voltage_mV: Supply voltage in millivolts.
        Bias_Current_mA: Laser bias current in milliamps.
        Interface_Error_Count: Cumulative interface error counter.
        Reboot_Count_Last_7D: Number of reboots in the last 7 days.
        Connected_Devices: Number of client devices currently connected.
        Device_Age_Days: Device age in days since manufacture / first activation.
        Maintenance_Count_Last_30D: Number of maintenance events in last 30 days.
        Failure_In_7_Days: Ground-truth label (1 = failure within 7 days, 0 = no failure).
    """

    model_config = {"str_strip_whitespace": True, "frozen": True}

    # Identifiers
    Device_ID: str = Field(..., min_length=1, description="Unique device identifier")
    Timestamp: str = Field(..., description="ISO-8601 formatted timestamp string")

    # Optical measurements
    Optical_RX_Power_dBm: OpticalPower = Field(
        ..., description="Received optical power (dBm)"
    )
    Optical_TX_Power_dBm: OpticalPower = Field(
        ..., description="Transmitted optical power (dBm)"
    )

    # Environmental / electrical
    Temperature_C: Temperature = Field(..., description="Device temperature (°C)")
    Voltage_mV: Voltage = Field(..., description="Supply voltage (mV)")
    Bias_Current_mA: BiasCurrent = Field(..., description="Laser bias current (mA)")

    # Operational counters
    Interface_Error_Count: NonNegativeInt = Field(
        ..., description="Cumulative interface error counter"
    )
    Reboot_Count_Last_7D: NonNegativeInt = Field(
        ..., description="Reboots in the last 7 days"
    )
    Connected_Devices: NonNegativeInt = Field(
        ..., description="Currently connected client devices"
    )
    Device_Age_Days: DeviceAge = Field(
        ..., description="Device age in days since first activation"
    )
    Maintenance_Count_Last_30D: NonNegativeInt = Field(
        ..., description="Maintenance events in the last 30 days"
    )

    # Target label
    Failure_In_7_Days: BinaryLabel = Field(
        ..., description="1 if the device failed within the next 7 days, else 0"
    )

    @field_validator("Timestamp")
    @classmethod
    def timestamp_must_be_parseable(cls, value: str) -> str:
        """Ensure *Timestamp* can be parsed as a datetime string.

        Args:
            value: Raw timestamp string from the incoming record.

        Returns:
            The original *value* if it is valid.

        Raises:
            ValueError: If the string cannot be parsed as a datetime.
        """
        import datetime  # local import to keep module-level imports lean

        try:
            datetime.datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"Timestamp '{value}' is not a valid ISO-8601 datetime string."
            ) from exc
        return value


# ---------------------------------------------------------------------------
# Batch-validation helper
# ---------------------------------------------------------------------------

def validate_records(
    raw_records: list[dict],
) -> tuple[list[TelemetryRecord], list[dict]):
    """Validate a list of raw telemetry dictionaries.

    Attempts to construct a :class:`TelemetryRecord` for every item in
    *raw_records*.  Invalid records are collected in the *rejected* list with
    a ``"_validation_error"`` key appended so that they can be audited later.

    Args:
        raw_records: List of dicts representing raw telemetry rows (e.g. from
            ``pandas.DataFrame.to_dict(orient="records")``).

    Returns:
        A two-tuple ``(valid, rejected)`` where *valid* is a list of
        :class:`TelemetryRecord` instances and *rejected* is a list of the
        original dicts that failed validation.

    Example:
        >>> valid, rejected = validate_records(df.to_dict(orient="records"))
        >>> print(f"Accepted {len(valid)}, dropped {len(rejected)}")
    """
    from pydantic import ValidationError

    valid: list[TelemetryRecord] = []
    rejected: list[dict] = []

    for record in raw_records:
        try:
            valid.append(TelemetryRecord.model_validate(record))
        except ValidationError as exc:
            logger.warning(
                "Validation failed for Device_ID=%s: %s",
                record.get("Device_ID", "<unknown>"),
                exc.error_count(),
            )
            rejected.append({**record, "_validation_error": str(exc)})

    logger.info(
        "Validation complete — accepted: %d, rejected: %d",
        len(valid),
        len(rejected),
    )
    return valid, rejected
