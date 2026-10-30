#!/usr/bin/env bash
set -euo pipefail

exec "${PYTHON:-python}" - "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" "$@" <<'PY'
import argparse
import csv
import math
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

runner_dir = Path(sys.argv[1])
sys.path.insert(0, str(runner_dir))
from npu_batch_sizes import NPU_BATCH_SIZES


def read_result(path, model, expected_batch):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1 or rows[0].get("name") != model or rows[0].get("dev") != "cuda":
        raise ValueError("expected exactly one CUDA result for this model")
    row = rows[0]
    if int(row["batch_size"]) != expected_batch:
        raise ValueError(f"batch size differs from NPU default {expected_batch}")
    if int(row["device_profile_steps"]) != 10:
        raise ValueError("expected 10 captured steps")
    fields = ("eager_device_latency", "compile_device_latency", "device_speedup")
    if not all(math.isfinite(float(row[key])) and float(row[key]) > 0 for key in fields):
        raise ValueError("missing or invalid device timings")
    return {
        "batch_size": row["batch_size"],
        "eager_device_ms_per_step": row["eager_device_latency"],
        "inductor_device_ms_per_step": row["compile_device_latency"],
        "device_speedup": row["device_speedup"],
        "profile_steps": row["device_profile_steps"],
    }


def main():
    parser = argparse.ArgumentParser(
        prog="run_selected_device_performance.sh",
        description="Run the selected 40 models sequentially on one CUDA GPU, inference then training."
    )
    parser.add_argument(
        "--device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        help="one physical GPU index or UUID; defaults to CUDA_VISIBLE_DEVICES, otherwise 0",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="new or empty results directory, relative to the calling directory",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="MODEL",
        help="space-separated exact model names from the 40-model list; defaults to all models",
    )
    args = parser.parse_args(sys.argv[2:])
    if not args.device.strip() or "," in args.device or args.device.strip() == "-1":
        parser.error("--device must select exactly one GPU")
    requested = set(args.models or ())
    known = {model for models in NPU_BATCH_SIZES.values() for model in models}
    unknown = requested - known
    if unknown:
        parser.error(f"unknown models: {', '.join(sorted(unknown))}; use exact names from the 40-model list")
    selected = {
        suite: {model: sizes for model, sizes in models.items() if not requested or model in requested}
        for suite, models in NPU_BATCH_SIZES.items()
    }
    output_dir = args.output_dir or Path(f"cuda_device_performance_{datetime.now():%Y%m%d_%H%M%S_%f}")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        parser.error(f"results directory must be new or empty: {output_dir}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.device.strip(), PYTHONUNBUFFERED="1")
    preflight = subprocess.run(
        [sys.executable, "-c", (
            "import torch; "
            "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
            "assert torch.cuda.device_count() == 1, 'Expected exactly one visible CUDA GPU'; "
            "print('PyTorch:', torch.__version__, 'GPU:', torch.cuda.get_device_name(0), flush=True)"
        )],
        cwd=runner_dir, env=env, check=False,
    )
    if preflight.returncode:
        return preflight.returncode
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    columns = [
        "order", "suite", "model", "mode", "batch_size", "eager_device_ms_per_step",
        "inductor_device_ms_per_step", "device_speedup", "profile_steps", "status", "exit_code",
        "message", "log_path", "profile_path",
    ]
    counts = {"ok": 0, "failed": 0, "unsupported": 0}
    total = 2 * sum(len(models) for models in selected.values())
    print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} (runner device: cuda:0)", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    with summary_path.open("x", newline="") as summary:
        writer = csv.DictWriter(summary, fieldnames=columns)
        writer.writeheader()
        summary.flush()
        order = 0
        for suite, models in selected.items():
            for model, (training_batch, inference_batch) in models.items():
                for mode, batch in (("inference", inference_batch), ("training", training_batch)):
                    order += 1
                    run_dir = output_dir / "runs" / f"{order:02d}_{suite}_{model}_{mode}"
                    run_dir.mkdir(parents=True)
                    log_path = run_dir / "run.log"
                    profile_path = run_dir / "profile"
                    result = dict(
                        order=order, suite=suite, model=model, mode=mode, status="unsupported",
                        message="", log_path=str(log_path), profile_path=str(profile_path),
                    )
                    print(f"\n[{order}/{total}] {suite}/{model} {mode}", flush=True)
                    if mode == "training" and (batch is None or model == "cm3leon_generate"):
                        result["message"] = (
                            "No training default in the pinned benchmark" if batch is None
                            else "Inference-only model; NPU converts training requests to inference"
                        )
                        log_path.write_text(result["message"] + "\n")
                        print(f"unsupported: {result['message']}", flush=True)
                    else:
                        overrides = {"CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"]}
                        if suite == "timm_models":
                            overrides.update(
                                HF_ENDPOINT="https://hf-mirror.com", TORCHBENCH_TIMM_SKIP_SSL_VERIFY="1",
                            )
                        run_env = dict(env, **overrides)
                        command = [
                            sys.executable, "-u", str(runner_dir / f"{suite}.py"),
                            "--performance", "--repeat", "1", "--iterations", "1", "--iterations-per-run", "1",
                            f"--{mode}", "--float32", "--device", "cuda", "--backend", "inductor",
                            "--disable-cudagraphs", "--npu-parity", "--only", model,
                            "--enable-profiler", "0", "--prof-output-path", str(profile_path), "--device-perf",
                            "--output", "result.csv", "--output-directory", str(run_dir),
                        ]
                        (run_dir / "command.sh").write_text(
                            "".join(f"export {key}={shlex.quote(value)}\n" for key, value in overrides.items())
                            + f"cd {shlex.quote(str(runner_dir))}\n{shlex.join(command)}\n"
                        )
                        with log_path.open("w") as log:
                            with subprocess.Popen(
                                command, cwd=runner_dir, env=run_env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace",
                            ) as process:
                                try:
                                    for line in process.stdout:
                                        log.write(line)
                                        log.flush()
                                        print(line, end="", flush=True)
                                    returncode = process.wait()
                                except KeyboardInterrupt:
                                    process.terminate()
                                    raise
                        result.update(status="failed", exit_code=returncode)
                        if returncode:
                            result["message"] = f"runner exited with code {returncode}; see run.log"
                        else:
                            try:
                                result.update(read_result(run_dir / "result.csv", model, batch), status="ok")
                            except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
                                result["message"] = f"invalid device result: {error}; see run.log"
                        print(f"status: {result['status']} {result['message']}", flush=True)
                    writer.writerow(result)
                    summary.flush()
                    counts[result["status"]] += 1
    print(f"\nFinished: {counts}; summary: {summary_path}", flush=True)
    return int(counts["failed"] > 0)


try:
    sys.exit(main())
except KeyboardInterrupt:
    print("\nInterrupted; completed rows are saved in summary.csv.", file=sys.stderr)
    sys.exit(130)
PY
