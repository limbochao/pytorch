import csv
import json
import shlex
import shutil
import subprocess
from pathlib import Path


class ArtifactCollector:
    def __init__(self, device_tag, csv_name):
        self.device_tag = device_tag
        self.csv_name = csv_name

    def collect(self, run_dir, artifact_dir, model, scene, copy=True, report=True):
        prefix = f"{model}_{scene}"
        manifest = run_dir / f"graphs_{self.device_tag}.json"
        sources = {}
        missing = {}
        repair_dir = run_dir / "repair"
        roots = (run_dir, repair_dir) if (repair_dir / "complete").is_file() else (run_dir,)
        for mode, label in (("eager", "Eager"), ("compile", "Inductor")):
            filename = f"{prefix}_{label}_{self.device_tag}.csv"
            if (artifact_dir / filename).is_file():
                sources[filename] = None
                continue
            for root in roots:
                candidates = sorted((root / "profile" / model / mode).rglob(self.csv_name))
                if len(candidates) == 1:
                    sources[filename] = candidates[0]
                    break
            else:
                missing[filename] = f"expected one {self.csv_name} in original or repair profile"

        names = ("output_code.py", "fx_graph_readable.py")
        for debug_dir in (root / "debug" for root in reversed(roots)):
            graphs = sorted({path.parent for name in names for path in debug_dir.rglob(name)})
            if graphs:
                break
        for index, graph in enumerate(graphs, 1):
            for name in names:
                filename = f"{prefix}_Inductor_{self.device_tag}_graph_{index:03d}_{graph.name}_{name}"
                if (artifact_dir / filename).is_file():
                    sources[filename] = None
                elif (graph / name).is_file():
                    sources[filename] = graph / name
                else:
                    missing[filename] = str(graph / name)
        if not graphs:
            codes = set(artifact_dir.glob(f"{prefix}_Inductor_{self.device_tag}_*_output_code.py"))
            readable = set(artifact_dir.glob(f"{prefix}_Inductor_{self.device_tag}_*_fx_graph_readable.py"))
            paired = {path.with_name(path.name.removesuffix("output_code.py") + "fx_graph_readable.py")
                      for path in codes}
            if manifest.is_file():
                for filename in json.loads(manifest.read_text()):
                    if (artifact_dir / filename).is_file():
                        sources[filename] = None
                    else:
                        missing[filename] = "saved graph artifact is missing and no source debug file remains"
            elif codes and paired == readable:
                sources.update({path.name: None for path in codes | readable})
            else:
                missing["output_code.py/fx_graph_readable.py"] = "no complete debug graph pairs found"

        if copy:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            for filename, source in sources.items():
                if source is not None:
                    target = artifact_dir / filename
                    temporary = target.with_suffix(target.suffix + ".tmp")
                    shutil.copy2(source, temporary)
                    temporary.replace(target)
            graph_files = sorted(name for name in sources if name.endswith(".py"))
            if graph_files and not any(name.endswith(".py") for name in missing):
                contents = json.dumps(graph_files, indent=2) + "\n"
                if not manifest.is_file() or manifest.read_text() != contents:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    temporary = manifest.with_suffix(".tmp")
                    temporary.write_text(contents)
                    temporary.replace(manifest)
        if report:
            print(f"[artifacts] {len(sources)} files -> {artifact_dir}", flush=True)
            for filename, reason in missing.items():
                print(f"[artifacts] missing: {filename}: {reason}", flush=True)
        return missing

    def repair(self, run_dir, artifact_dir, model, scene, batch, command, env, overrides):
        missing = self.collect(run_dir, artifact_dir, model, scene)
        if not missing:
            return True
        needs_profile = any(name.endswith(".csv") for name in missing)
        work_dir = run_dir / "repair"
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "complete").unlink(missing_ok=True)
        for stale in work_dir.glob("result*"):
            if stale.is_file():
                stale.unlink()
        for name in ("profile", "debug", "cache"):
            if (work_dir / name).exists():
                shutil.rmtree(work_dir / name)
        repair_command = command[:]
        for flag in ("--enable-profiler", "--prof-output-path", "--output", "--output-directory", "--batch-size"):
            if flag in repair_command:
                index = repair_command.index(flag)
                del repair_command[index:index + 2]
        if "--device-perf" in repair_command:
            repair_command.remove("--device-perf")
        repair_command += ["--batch-size", str(batch), "--output", "result.csv",
                           "--output-directory", str(work_dir)]
        if needs_profile:
            repair_command += ["--enable-profiler", "0", "--device-perf",
                               "--prof-output-path", str(work_dir / "profile")]
        else:
            repair_command.append("--compile-only")
        repair_overrides = dict(
            overrides,
            TORCH_COMPILE_DEBUG_DIR=str(work_dir / "debug"),
            TORCHINDUCTOR_CACHE_DIR=str(work_dir / "cache" / "inductor"),
            TRITON_CACHE_DIR=str(work_dir / "cache" / "triton"),
        )
        (work_dir / "command.sh").write_text(
            "".join(f"export {key}={shlex.quote(value)}\n" for key, value in repair_overrides.items())
            + f"cd {shlex.quote(str(Path(command[2]).parent))}\n{shlex.join(repair_command)}\n"
        )
        print(f"[artifacts] {'profile missing files' if needs_profile else 'compile missing graphs'}: {model} {scene}",
              flush=True)
        with (work_dir / "run.log").open("w") as log:
            with subprocess.Popen(
                repair_command, cwd=Path(command[2]).parent, env=dict(env, **repair_overrides),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
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
        if returncode:
            print(f"[artifacts] repair failed ({returncode}); original performance retained; "
                  f"see {work_dir / 'run.log'}", flush=True)
            return False
        if not needs_profile and f"[compile-only] batch_size={batch}\n" not in (work_dir / "run.log").read_text():
            print("[artifacts] repair did not confirm the saved batch size; original performance retained", flush=True)
            return False
        if needs_profile:
            with (work_dir / "result.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != 1 or rows[0].get("name") != model or int(rows[0]["batch_size"]) != int(batch):
                raise ValueError("repair profile does not match the saved model and batch size")
        names = ("output_code.py", "fx_graph_readable.py")
        graphs = {path.parent for name in names for path in (work_dir / "debug").rglob(name)}
        if not graphs or any(not (graph / name).is_file() for graph in graphs for name in names):
            print(f"[artifacts] repair produced incomplete graphs; see {work_dir / 'run.log'}", flush=True)
            return False
        prefix = f"{model}_{scene}_Inductor_{self.device_tag}_"
        for pattern in (prefix + "*.py", prefix + "*.py.tmp"):
            for stale in artifact_dir.glob(pattern):
                if stale.is_file():
                    stale.unlink()
        (work_dir / "complete").write_text(str(batch) + "\n")
        return not self.collect(run_dir, artifact_dir, model, scene)
