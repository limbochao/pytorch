#!/usr/bin/env bash
set -euo pipefail

exec "${PYTHON:-python}" - "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" "$@" <<'PY'
import argparse
import csv
import math
import os
import shlex
import shutil
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
    fields = ("eager_device_latency", "compile_device_latency", "device_speedup")
    if not all(row.get(key) for key in (*fields, "device_profile_steps")):
        raise ValueError("runner produced no device profile result; check the original traceback in run.log")
    if int(row["batch_size"]) != expected_batch:
        raise ValueError(f"batch size differs from NPU default {expected_batch}")
    if int(row["device_profile_steps"]) != 10:
        raise ValueError("expected 10 captured steps")
    if not all(math.isfinite(float(row[key])) and float(row[key]) > 0 for key in fields):
        raise ValueError("missing or invalid device timings")
    return {
        "batch_size": row["batch_size"],
        "eager_device_ms_per_step": row["eager_device_latency"],
        "inductor_device_ms_per_step": row["compile_device_latency"],
        "device_speedup": row["device_speedup"],
        "profile_steps": row["device_profile_steps"],
    }


def write_summary(path, columns, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def collect_artifacts(run_dir, artifact_dir, model, scene):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    sources = {}
    missing = []
    prefix = f"{model}_{scene}"
    for mode, label in (("eager", "Eager"), ("compile", "Inductor")):
        candidates = sorted((run_dir / "profile" / model / mode).rglob("op_statistics.csv"))
        filename = f"{prefix}_{label}_GPU.csv"
        if len(candidates) == 1:
            sources[filename] = candidates[0]
        else:
            missing.append(f"{filename}: expected one op_statistics.csv, found {len(candidates)}")
    names = ("output_code.py", "fx_graph_readable.py")
    debug_dir = run_dir / "debug"
    graphs = sorted({path.parent for name in names for path in debug_dir.rglob(name)})
    if not graphs:
        missing.append("Inductor output_code.py and fx_graph_readable.py: no debug graphs found")
    for index, graph in enumerate(graphs, 1):
        for name in names:
            filename = f"{prefix}_Inductor_GPU_graph_{index:03d}_{graph.name}_{name}"
            source = graph / name
            if source.is_file():
                sources[filename] = source
            else:
                missing.append(f"{graph.relative_to(debug_dir)}/{name}")
    for filename, source in sources.items():
        target = artifact_dir / filename
        if not target.is_file():
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copy2(source, temporary)
            temporary.replace(target)
    print(f"[artifacts] {len(sources)} files -> {artifact_dir}", flush=True)
    for item in missing:
        print(f"[artifacts] missing: {item}", flush=True)


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
        help="results directory; reuse it to retry failed or unfinished cases",
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
    summary_path = output_dir / "summary.csv"
    columns = [
        "order", "suite", "model", "mode", "batch_size", "eager_device_ms_per_step",
        "inductor_device_ms_per_step", "device_speedup", "profile_steps", "status", "exit_code",
        "message", "log_path", "profile_path",
    ]
    rows = []
    for suite, models in selected.items():
        for model in models:
            for mode in ("inference", "training"):
                order = len(rows) + 1
                run_dir = output_dir / "runs" / f"{order:02d}_{suite}_{model}_{mode}"
                result = dict.fromkeys(columns, "")
                result.update(
                    order=order, suite=suite, model=model, mode=mode, status="pending",
                    log_path=str(run_dir / "run.log"), profile_path=str(run_dir / "profile"),
                )
                rows.append(result)
    if output_dir.exists() and not output_dir.is_dir():
        parser.error(f"results path must be a directory: {output_dir}")
    if summary_path.exists():
        try:
            with summary_path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                previous = list(reader)
                if reader.fieldnames != columns:
                    raise ValueError("unexpected summary columns")
            keys = ("order", "suite", "model", "mode")
            expected = [tuple(str(row[key]) for key in keys) for row in rows]
            saved = [tuple(row[key] for key in keys) for row in previous]
            if saved != expected[:len(saved)]:
                raise ValueError("model selection differs from the saved summary; reuse the original --models")
            for row, saved_row in zip(rows, previous):
                if None in saved_row or any(value is None for value in saved_row.values()):
                    raise ValueError("incomplete summary row")
                if saved_row["status"] not in {"ok", "failed", "unsupported", "pending"}:
                    raise ValueError("unknown saved status")
                row.update(saved_row)
        except (OSError, ValueError, KeyError, csv.Error) as error:
            parser.error(f"cannot resume {summary_path}: {error}")
    elif output_dir.exists() and any(path.name != "summary.tmp" for path in output_dir.iterdir()):
        parser.error(f"nonempty results directory has no summary.csv: {output_dir}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.device.strip(), PYTHONUNBUFFERED="1")
    if any(row["status"] not in {"ok", "unsupported"} for row in rows):
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
    write_summary(summary_path, columns, rows)
    counts = {"ok": 0, "failed": 0, "unsupported": 0}
    print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} (runner device: cuda:0)", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    for result in rows:
        order, suite, model, mode = (result[key] for key in ("order", "suite", "model", "mode"))
        run_dir = output_dir / "runs" / f"{int(order):02d}_{suite}_{model}_{mode}"
        scene = "Train" if mode == "training" else "Infer"
        artifact_dir = output_dir / "perf" / f"{model}_{scene}"
        print(f"\n[{order}/{len(rows)}] {suite}/{model} {mode}", flush=True)
        if result["status"] in {"ok", "unsupported"}:
            print(f"skip: saved {result['status']}", flush=True)
            if result["status"] == "ok":
                try:
                    collect_artifacts(run_dir, artifact_dir, model, scene)
                except OSError as error:
                    print(f"[artifacts] collection failed: {error}", flush=True)
            counts[result["status"]] += 1
            continue
        log_path = run_dir / "run.log"
        profile_path = run_dir / "profile"
        result.update(dict.fromkeys((
            "batch_size", "eager_device_ms_per_step", "inductor_device_ms_per_step", "device_speedup",
            "profile_steps", "exit_code", "message",
        ), ""))
        result.update(status="pending", log_path=str(log_path), profile_path=str(profile_path))
        write_summary(summary_path, columns, rows)
        run_dir.mkdir(parents=True, exist_ok=True)
        for stale in run_dir.glob("result*"):
            if stale.is_file():
                stale.unlink()
        for stale in (profile_path, run_dir / "debug", run_dir / "cache"):
            if stale.exists():
                shutil.rmtree(stale)
        for pattern in (f"{model}_{scene}_Eager_GPU.csv", f"{model}_{scene}_Inductor_GPU.csv",
                        f"{model}_{scene}_Inductor_GPU_*.py"):
            for stale in (*artifact_dir.glob(pattern), *artifact_dir.glob(pattern + ".tmp")):
                if stale.is_file():
                    stale.unlink()
        batch = selected[suite][model][0 if mode == "training" else 1]
        if mode == "training" and (batch is None or model == "cm3leon_generate"):
            result.update(status="unsupported", message=(
                "No training default in the pinned benchmark" if batch is None
                else "Inference-only model; NPU converts training requests to inference"
            ))
            log_path.write_text(result["message"] + "\n")
            print(f"unsupported: {result['message']}", flush=True)
        else:
            overrides = {
                "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
                "TORCH_COMPILE_DEBUG": "1",
                "TORCH_COMPILE_DEBUG_DIR": str(run_dir / "debug"),
                "TORCHINDUCTOR_CACHE_DIR": str(run_dir / "cache" / "inductor"),
                "TRITON_CACHE_DIR": str(run_dir / "cache" / "triton"),
                "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1",
            }
            if suite == "timm_models":
                overrides.update(
                    HF_ENDPOINT=env.get("HF_ENDPOINT") or "https://hf-mirror.com",
                    TORCHBENCH_TIMM_SKIP_SSL_VERIFY="1",
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
        write_summary(summary_path, columns, rows)
        if result["status"] != "unsupported":
            try:
                collect_artifacts(run_dir, artifact_dir, model, scene)
            except OSError as error:
                print(f"[artifacts] collection failed: {error}", flush=True)
        counts[result["status"]] += 1
    print(f"\nFinished: {counts}; summary: {summary_path}", flush=True)
    return int(counts["failed"] > 0)


try:
    sys.exit(main())
except KeyboardInterrupt:
    print("\nInterrupted; completed rows are saved in summary.csv.", file=sys.stderr)
    sys.exit(130)
PY
