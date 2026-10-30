import math
import tempfile
from pathlib import Path


try:
    from .cuda_op_statistic import CudaOpStatisticParser
except ImportError:
    from cuda_op_statistic import CudaOpStatisticParser


def read_kernel_duration(trace_path, steps):
    """Return cumulative kernel duration per captured model step, in milliseconds."""
    if steps <= 0:
        raise RuntimeError(f"Invalid captured step count: {steps}")
    total_us = math.fsum(item.total_us for item in CudaOpStatisticParser(trace_path).statistics())
    if total_us <= 0:
        raise RuntimeError(f"No positive CUDA kernel duration in {trace_path}; check profiler/CUPTI support")
    return total_us / steps / 1000


def profile_cuda_model(fn, save_path):
    """Capture ten warmed-up model steps separately from end-to-end timing."""
    import torch

    warmup = 10
    active = 10
    Path(save_path).mkdir(parents=True, exist_ok=True)
    trace_dir = Path(tempfile.mkdtemp(prefix="performance_", dir=save_path))
    trace_path = trace_dir / "trace.json"
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=warmup, active=active, repeat=1),
        on_trace_ready=lambda prof: prof.export_chrome_trace(str(trace_path)),
    ) as prof:
        for _ in range(warmup + active):
            fn()
            torch.cuda.synchronize()
            prof.step()
    CudaOpStatisticParser(trace_path).export_csv(trace_dir / "op_statistics.csv")
    return str(trace_dir), active
