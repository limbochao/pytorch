"""Convert a PyTorch CUDA Chrome Trace into an ``op_statistics.csv`` file.

CUDA profiler traces contain device-side kernel events and CPU operator events.
Both carry the same ``External id`` for a launched operation, so this tool
aggregates kernel durations by the associated CPU operator.  The result uses
the column layout of torch_npu's ``op_statistic.csv`` while retaining CUDA
specific values in ``Core Type``.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CSV_COLUMNS = (
    "Device_id",
    "OP Type",
    "Core Type",
    "Count",
    "Total Time(us)",
    "Min Time(us)",
    "Avg Time(us)",
    "Max Time(us)",
    "Ratio(%)",
)


@dataclass(frozen=True)
class OperationStatistic:
    """Aggregated CUDA kernel durations for one operation and device."""

    device_id: int
    op_type: str
    count: int
    total_us: float
    min_us: float
    avg_us: float
    max_us: float
    ratio: float


class CudaOpStatisticParser:
    """Parse CUDA kernel events and aggregate them in NPU statistic format."""

    def __init__(
        self,
        trace_path: str | Path,
        *,
        core_type: str = "CUDA_CORE",
        device_id_base: int = 1,
        name_by: str = "operator",
        strip_aten_prefix: bool = True,
    ) -> None:
        if name_by not in {"operator", "kernel"}:
            raise ValueError("name_by must be 'operator' or 'kernel'")
        self.trace_path = Path(trace_path)
        self.core_type = core_type
        self.device_id_base = device_id_base
        self.name_by = name_by
        self.strip_aten_prefix = strip_aten_prefix

    def statistics(self) -> list[OperationStatistic]:
        """Return statistics sorted by total device time, descending."""

        events = self._trace_events()
        operators = self._operator_names(events)
        groups: dict[tuple[int, str], list[float]] = defaultdict(list)

        for event in events:
            if event.get("ph") != "X" or event.get("cat") != "kernel":
                continue
            duration = event.get("dur")
            if (
                not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or not math.isfinite(duration)
                or duration < 0
            ):
                raise ValueError(f"Invalid CUDA kernel duration in {self.trace_path}: {duration!r}")

            args = event.get("args")
            if not isinstance(args, Mapping):
                args = {}
            device = args.get("device", 0)
            try:
                device_id = int(device) + self.device_id_base
            except (TypeError, ValueError):
                device_id = self.device_id_base

            if self.name_by == "kernel":
                op_name = event.get("name")
            else:
                op_name = operators.get(self._external_id(args))
                if not op_name:
                    op_name = event.get("name")
            if not isinstance(op_name, str) or not op_name:
                continue
            groups[(device_id, op_name)].append(float(duration))

        total_time = math.fsum(math.fsum(values) for values in groups.values())
        result = []
        for (device_id, op_name), values in groups.items():
            total = math.fsum(values)
            result.append(
                OperationStatistic(
                    device_id=device_id,
                    op_type=op_name,
                    count=len(values),
                    total_us=total,
                    min_us=min(values),
                    avg_us=total / len(values),
                    max_us=max(values),
                    ratio=total * 100.0 / total_time if total_time else 0.0,
                )
            )
        result.sort(key=lambda item: (-item.total_us, item.device_id, item.op_type))
        return result

    def export_csv(self, output_path: str | Path) -> Path:
        """Write an NPU-compatible statistic CSV and return its path."""

        csv_path = Path(output_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for item in self.statistics():
                writer.writerow(
                    {
                        "Device_id": item.device_id,
                        "OP Type": item.op_type,
                        "Core Type": self.core_type,
                        "Count": item.count,
                        "Total Time(us)": _format_float(item.total_us),
                        "Min Time(us)": _format_float(item.min_us),
                        "Avg Time(us)": _format_float(item.avg_us),
                        "Max Time(us)": _format_float(item.max_us),
                        "Ratio(%)": _format_float(item.ratio),
                    }
                )
        return csv_path

    def _trace_events(self) -> list[Mapping[str, Any]]:
        with self.trace_path.open(encoding="utf-8") as trace_file:
            trace = json.load(trace_file)
        events = trace.get("traceEvents") if isinstance(trace, Mapping) else trace
        if not isinstance(events, list):
            raise ValueError(f"{self.trace_path} does not contain a traceEvents list")
        return [event for event in events if isinstance(event, Mapping)]

    def _operator_names(self, events: list[Mapping[str, Any]]) -> dict[str, str]:
        candidates: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for event in events:
            if event.get("ph") != "X" or event.get("cat") != "cpu_op":
                continue
            args = event.get("args")
            if not isinstance(args, Mapping):
                continue
            external_id = self._external_id(args)
            name = event.get("name")
            if external_id and isinstance(name, str) and name:
                candidates[external_id].append(event)

        names = {}
        for external_id, matching_events in candidates.items():
            # If a trace contains duplicate IDs, prefer the innermost CPU op.
            event = min(matching_events, key=lambda item: float(item.get("dur", float("inf"))))
            name = str(event["name"])
            if self.strip_aten_prefix:
                name = name.removeprefix("aten::")
            names[external_id] = name
        return names

    @staticmethod
    def _external_id(args: Mapping[str, Any]) -> str:
        value = args.get("External id")
        return "" if value is None else str(value)


def _format_float(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")
