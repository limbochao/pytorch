# CUDA/NPU TorchBench device performance comparison

中文部署步骤、两端参数对照与结果核对方法见
[CUDA 环境部署与 NPU 对齐测试指南](torchbench_cuda_npu_deployment_zh.md)。

This test branch starts from PyTorch `v2.13.0`. Use a CUDA-enabled PyTorch 2.13.0
environment with its matching Triton, torchvision and torchaudio packages.
Running these Python benchmark scripts does not require rebuilding PyTorch.

## Use the same model source

Pin `pytorch/benchmark` to **`fb14629994956c16d27c7a9940c5f081afbf68dc`**,
the revision specified by the
[NPU TorchBench README](https://gitcode.com/luqichao/pytorch_new/blob/master/benchmarks/torchbench/README.md).
Do not use the latest benchmark checkout for a comparison against that NPU setup.

From this PyTorch checkout:

```bash
cd benchmarks/dynamo
git clone --depth=1 https://github.com/pytorch/benchmark.git benchmark
git -C benchmark fetch --depth=1 origin fb14629994956c16d27c7a9940c5f081afbf68dc
git -C benchmark checkout --detach fb14629994956c16d27c7a9940c5f081afbf68dc
python -m pip install numpy pandas psutil pyyaml scipy tqdm
python benchmark/install.py dlrm
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
```

If a benchmark checkout already exists, use it instead of cloning another one,
and check its revision with `git -C benchmark rev-parse HEAD`.
The runner finds `./benchmark` when launched from `benchmarks/dynamo`.
For models that use TorchBench's `DATA_PATH`, set `TORCHBENCH_DATA_PATH` to the
CUDA machine's dataset directory. Use the same data as in the NPU run.

## DLRM commands

Run from `benchmarks/dynamo` with an available CUDA device. These commands match
the previously tested NPU batch sizes: inference 2048 and training 1024, FP32,
static shapes, and graph capture disabled.

```bash
# Inference
CUDA_VISIBLE_DEVICES=0 python torchbench.py \
  --device cuda --only dlrm \
  --performance --inference --backend inductor \
  --float32 --batch-size 2048 --npu-parity --disable-cudagraphs \
  --repeat 30 --enable-profiler 0 --prof-output-path ./profile_cuda_inference \
  --device-perf --output ./dlrm_cuda_inference.csv --output-directory "$PWD"

# Training
CUDA_VISIBLE_DEVICES=0 python torchbench.py \
  --device cuda --only dlrm \
  --performance --training --backend inductor \
  --float32 --batch-size 1024 --npu-parity --disable-cudagraphs \
  --repeat 30 --enable-profiler 0 --prof-output-path ./profile_cuda_training \
  --device-perf --output ./dlrm_cuda_training.csv --output-directory "$PWD"
```

Use the matching dataset, dtype, input shapes and batch size for other models.
CUDA uses its own Inductor/Triton implementation; omit NPU-specific arguments
such as `--npu-backend` and replace `--disable-aclgraph` with `--disable-cudagraphs`.

`--npu-parity` disables CUDA TF32. For training, it matches the NPU runner's
SGD optimizer (`lr=0.01`, `foreach=False`), allows Dynamo to trace zero-grad and
optimizer step, and returns the scalar loss when outputs are not collected.
Without this option, the CUDA runner keeps its community optimizer policy.
Device-specific NPU patches and CPU fallbacks are not reproduced on CUDA.

## Sampling and calculation

`--enable-profiler` enables separate eager and compile profile runs after the
existing end-to-end benchmark. Its optional value is `0` for command-line
compatibility with the NPU test. It requires CUDA performance mode and Inductor.
`--device-perf` additionally analyzes the captured traces.

Each mode executes **10 warmup steps followed by 10 captured steps**. The
schedule is fixed, independent of `--repeat`, `--iterations` and
`--iterations-per-run`. Each step calls the same model iteration function used
by the end-to-end benchmark and then synchronizes CUDA. Warmup and first-time
compilation are excluded from the captured steps.

Raw Chrome traces are saved independently for each run:

```text
<prof-output-path>/<model>/eager/performance_*/trace.json
<prof-output-path>/<model>/eager/performance_*/op_statistics.csv
<prof-output-path>/<model>/compile/performance_*/trace.json
<prof-output-path>/<model>/compile/performance_*/op_statistics.csv
```

The duration parser follows `torch-learn/scripts/tools/cuda_op_statistic.py`:
select only complete events with `ph="X"` and `cat="kernel"`, then sum their
`dur` fields, which are in microseconds. CPU operators, CUDA runtime events,
memory copies and memset events are excluded. The parser associates kernels
with CPU operator names using `External id`, falling back to the kernel name
when no operator matches. There is no `kernel_details.csv` on CUDA and none is
generated.

Each trace directory also contains `op_statistics.csv`, with the same columns
as the reference tool: `Device_id`, `OP Type`, `Core Type`, `Count`,
`Total Time(us)`, `Min Time(us)`, `Avg Time(us)`, `Max Time(us)` and `Ratio(%)`.
`Core Type` is `CUDA_CORE`, and displayed device IDs start at 1. The device
latency calculation uses unrounded durations from the trace, not the rounded
CSV values.

```text
eager_device_latency   = sum(eager kernel dur) / 10 / 1000  # ms/step
compile_device_latency = sum(compile kernel dur) / 10 / 1000
device_speedup         = eager_device_latency / compile_device_latency
```

Overlapping kernels on different streams are counted separately. This measures
cumulative kernel execution time, not elapsed device time or end-to-end time.
The formula matches the NPU calculation from `kernel_details.csv` durations.
Missing or invalid kernel durations cause an error instead of a device result;
an empty kernel trace can indicate unavailable CUDA profiler/CUPTI support.

With `--device-perf`, the final result shows eager and compile device latency
and their ratio, without the end-to-end ratio or latency:

```text
[device] eager=2.000000 ms/step, compile=1.000000 ms/step, device_speedup=2.000x
```

The summary CSV retains the existing end-to-end `speedup` and `abs_latency`
columns and adds `device_speedup`, `eager_device_latency`,
`compile_device_latency` and `device_profile_steps` (10). `--output` selects
this summary file; it does not select the raw trace location.

Omit `--device-perf` to capture profiles while keeping the normal end-to-end
speedup output. Omit both profiling flags for an ordinary performance run.
