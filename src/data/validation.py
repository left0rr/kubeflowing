"""
Pydantic validation models for GPON router TR-069 telemetry rows.

Each incoming telemetry record is validated against TelemetryRecord
before entering the feature-engineering stage. Invalid rows are logged and
dropped rather than raising hard errors so that a single bad record cannot
stall a batch ingestion job.
"""

from __future__ import annotations

import logging
from typing import Annotated

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

OpticalPower = Annotated[float, Field(ge=-40.0, le=10.0)]
Temperature = Annotated[float, Field(ge=-40.0, le=85.0)]
Voltage = Annotated[float, Field(ge=2_500.0, le=4_000.0)]
BiasCurrent = Annotated[float, Field(ge=0.0, le=100.0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
DeviceAge = Annotated[int, Field(ge=1)]
BinaryLabel = Annotated[int, Field(ge=0, le=1)]


class TelemetryRecord(BaseModel):
    model_config = {"str_strip_whitespace": True, "frozen": True}

    Device_ID: str = Field(..., min_length=1)
    Timestamp: str = Field(...)
    Optical_RX_Power_dBm: OpticalPower
    Optical_TX_Power_dBm: OpticalPower
    Temperature_C: Temperature
    Voltage_mV: Voltage
    Bias_Current_mA: BiasCurrent
    Interface_Error_Count: NonNegativeInt
    Reboot_Count_Last_7D: NonNegativeInt
    Connected_Devices: NonNegativeInt
    Device_Age_Days: DeviceAge
    Maintenance_Count_Last_30D: NonNegativeInt
    Failure_In_7_Days: BinaryLabel

    @field_validator("Timestamp")
    @classmethod
    def timestamp_must_be_parseable(cls, value: str) -> str:
        import datetime
        try:
            datetime.datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"Timestamp '{value}' is not a valid ISO-8601 datetime string.") from exc
        return value


def validate_records(raw_records: list[dict]) -> tuple[list[TelemetryRecord], list[dict]]:
    from pydantic import ValidationError
    valid: list[TelemetryRecord] = []
    rejected: list[dict] = []
    for record in raw_records:
        try:
            valid.append(TelemetryRecord.model_validate(record))
        except ValidationError as exc:
            logger.warning("Validation failed for Device_ID=%s: %s errors", record.get("Device_ID", "<unknown>"), exc.error_count())
            rejected.append({**record, "_validation_error": str(exc)})
    logger.info("Validation complete — accepted: %d, rejected: %d", len(valid), len(rejected))
    return valid, rejected
