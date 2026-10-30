# CUDA 环境部署与 NPU 对齐测试指南

本文用于在 CUDA 环境部署本次 TorchBench 改动，比较同一个模型在各自设备上
`Inductor + Triton` 相对于 eager 的 device 侧收益。以此前 NPU 测试的 DLRM 为例，
给出版本、配置、运行命令和结果核对方法。

实现已经完成本地解析和模拟调用检查，尚未经过 CUDA 真机验证。

## 1. 固定代码版本与测试配置

CUDA 使用个人 fork 的 `test/cuda-device-perf` 分支，基线为 PyTorch `v2.13.0`，
profile 功能起始提交为 `a234bc3d88c6fa43b83bc299eb7f0e4dd440097b`；
本文的默认 batch 同步需使用该分支的最新版本。
NPU 对照使用 `pytorch_new` 的 `test/cicd_benchmark` 分支，
参考实现提交为 `9fb9ca72e8cf`，其中包含 `--device-perf` 和固定 10 轮采集功能。

**两端模型代码均固定为 `pytorch/benchmark` 的以下提交：**

```text
fb14629994956c16d27c7a9940c5f081afbf68dc
```

该提交来自 [NPU TorchBench README](https://gitcode.com/luqichao/pytorch_new/blob/master/benchmarks/torchbench/README.md)。
不要将 NPU 的这个版本与 CUDA 上最新的 benchmark 模型直接混用。

| 对齐项目 | NPU | CUDA |
| --- | --- | --- |
| runner 入口 | `benchmarks/torchbench/torchbench.py` | `benchmarks/dynamo/torchbench.py` |
| benchmark 模型提交 | 上述固定提交 | 同一个提交 |
| 推理场景 | `--inference --batch-size 2048` | 相同 |
| 训练场景 | `--training --batch-size 1024` | 相同 |
| 精度 | `--float32` | `--float32 --npu-parity`，禁用 CUDA TF32 |
| 编译后端 | `--backend inductor --npu-backend triton` | `--backend inductor` |
| 图捕获 | `--disable-aclgraph` | `--disable-cudagraphs` |
| 输入形状 | 静态，不加动态 shape 参数 | 相同，且实际 shape、stride 和输入配置应匹配 |
| 训练优化器 | SGD，`lr=0.01`，`foreach=False` | 通过 `--npu-parity` 对齐 |
| profile 采样 | 每种模式预热 10 轮，再采集 10 轮 | 相同 |
| 端到端测量 | `--repeat 30 --iterations-per-run 1` | 相同 |
| device 分析开关 | `--enable-profiler 0 --device-perf` | 相同 |

表中的 batch size 专用于本次 DLRM 对照。其他模型也需对齐序列长度、图像尺寸等输入配置，
不能统一套用 DLRM 的数值。如果 NPU 测试手动改变了 batch size，应在 CUDA 中显式匹配。

### 默认 batch size 如何同步

performance 模式启用 `--npu-parity` 后，默认 batch **只对指定的 40 个模型同步**：
HuggingFace 18 个、TIMM 6 个、TorchBench 16 个。
完整名单、训练和推理 batch 及三类 runner 示例见
[选定模型的 NPU performance batch 对照表](npu_performance_batch_sizes_zh.md)。

默认值集中在 [npu_batch_sizes.py](npu_batch_sizes.py)，取自当前 NPU runner 的实际规则，
包括 HF 除数及 MT5 的固定 batch 4、TIMM 列表、TorchBench 默认值和 NPU batch 补丁。
清单外模型保持社区默认值，不会一并改为 NPU 默认。
允许自定义 batch 的模型仍优先使用显式 `--batch-size` 或 batch-size-file 传入值。
accuracy 模式和不加 `--npu-parity` 的运行也保持原有 batch 选择。

DLRM 默认推理为 **2048**、训练为 **1024**，后文 CUDA 命令不需要手写 batch size。
这避免了某些 GPU 上 DLRM 被 metadata 改为推理 batch size 1 的情况。
`soft_actor_critic` 的 NPU 补丁将默认 batch 改为 **32768**，CUDA 也只同步这项 batch 设置，
不移植该模型的其他 NPU 算子补丁。

NPU 某些模型会强制覆盖显式 batch，例如 MT5、dcgan、lennard_jones 和
nvidia_deeprecommender。CUDA 对允许自定义 batch 的模型尊重显式设置，
修改时应核对两端 CSV 的实际 `batch_size`，不能只比较命令参数。
`llama` 在固定版本中没有训练默认 batch；`cm3leon_generate` 在 NPU 中会把训练请求转为推理，
不能将其算作有效训练性能。此次默认值同步不扩展模型的训练能力。

## 2. 准备 CUDA Python 环境

在 CUDA 测试容器或既有 CUDA Python 环境中操作。先准备好 CUDA 版 PyTorch 2.13.0、
配套的 torchvision、torchaudio 和 CUDA Triton；具体 CUDA 包应与测试机器驱动匹配。
本次修改均为 benchmark 的 Python 文件，使用已安装的 PyTorch 即可，不需要重新编译 PyTorch。

CUDA 环境使用 CUDA 版本的 Triton，不沿用 NPU 环境的 `triton-ascend` 或
`transfer_to_npu` 启动配置。

先确认设备和 Python 实际加载的包。包检查应从 PyTorch 源码根目录之外执行，
例如后文的 `benchmarks/dynamo`，避免源码目录遮蔽已安装的 `torch`：

```bash
nvidia-smi
python - <<'PY'
import torch
import torchvision
import torchaudio
import triton

print("torch:", torch.__version__, torch.__file__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
print("triton:", triton.__version__, triton.__file__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
assert torch.version.cuda is not None and torch.cuda.is_available()
assert torch.__version__.split("+", 1)[0] == "2.13.0"
print("visible device 0:", torch.cuda.get_device_name(0))
PY
```

如果采用其他 PyTorch 版本，需要单独注明版本差异，不将它视为本次 `v2.13.0` 基线结果。
NPU 侧也应记录实际安装的 `torch`、`torch_npu`、Triton 和 CANN 版本；
runner 基线和安装包版本是两项信息，都需要保留。

## 3. 获取 CUDA runner 与同版本模型

以下命令在 Bash 中执行。已有 checkout 时，在确认没有待保留修改后切换到对应分支，
不需要重复 clone。

```bash
git clone --branch test/cuda-device-perf --single-branch \
  https://github.com/limbochao/pytorch.git pytorch_cuda
cd pytorch_cuda
git fetch origin tag v2.13.0
git log -1 --oneline
git merge-base HEAD v2.13.0

cd benchmarks/dynamo
git clone --depth=1 https://github.com/pytorch/benchmark.git benchmark
git -C benchmark fetch --depth=1 origin fb14629994956c16d27c7a9940c5f081afbf68dc
git -C benchmark checkout --detach fb14629994956c16d27c7a9940c5f081afbf68dc
git -C benchmark rev-parse HEAD

python -m pip install numpy pandas psutil pyyaml scipy tqdm
python benchmark/install.py dlrm
```

`git merge-base` 的预期结果为 `cf30153c4c131c8164ee7798e5022d810682e2cb`，
即本次 `v2.13.0` 基线；benchmark 的 `rev-parse HEAD` 必须与固定提交一致。
模型依赖安装后，再执行上一节的包版本检查，并用 `python -m pip freeze` 保留依赖清单。

运行入口是当前目录下的 `torchbench.py`，不要换成 benchmark 仓库自己的模型运行入口。
runner 会加载这里的 `benchmark/torchbenchmark` 模型。
对于使用 TorchBench `DATA_PATH` 的模型，可设置数据集的绝对路径：

```bash
export TORCHBENCH_DATA_PATH=/absolute/path/to/torchbench_data
```

数据内容应与 NPU 相同，目录路径可以不同。此变量只指定数据位置，不负责下载数据；
使用模型自身固定数据路径的场景，还需按该模型要求准备数据。

## 4. CUDA 测试命令

下面两组命令均在 `pytorch_cuda/benchmarks/dynamo` 执行。
`CUDA_VISIBLE_DEVICES=0` 是示例，替换为已确认空闲的 GPU。
结果目录使用绝对路径，分别保存推理和训练产物，避免相对路径或旧 profile 干扰核对。

### 4.1 DLRM FP32 静态推理

```bash
set -o pipefail
cuda_infer_dir="$PWD/results/dlrm_inference_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$cuda_infer_dir"

CUDA_VISIBLE_DEVICES=0 python torchbench.py \
  --device cuda --only dlrm \
  --performance --inference --backend inductor \
  --float32 --npu-parity --disable-cudagraphs \
  --repeat 30 --iterations-per-run 1 \
  --enable-profiler 0 --prof-output-path "$cuda_infer_dir/profile" \
  --device-perf --output result.csv --output-directory "$cuda_infer_dir" \
  2>&1 | tee "$cuda_infer_dir/run.log"
```

### 4.2 DLRM FP32 静态训练

```bash
set -o pipefail
cuda_train_dir="$PWD/results/dlrm_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$cuda_train_dir"

CUDA_VISIBLE_DEVICES=0 python torchbench.py \
  --device cuda --only dlrm \
  --performance --training --backend inductor \
  --float32 --npu-parity --disable-cudagraphs \
  --repeat 30 --iterations-per-run 1 \
  --enable-profiler 0 --prof-output-path "$cuda_train_dir/profile" \
  --device-perf --output result.csv --output-directory "$cuda_train_dir" \
  2>&1 | tee "$cuda_train_dir/run.log"
```

`--npu-parity` 除了同步清单内模型的 performance 默认 batch，还匹配当前 NPU TorchBench runner 的训练迭代行为：

- 使用普通 SGD，学习率为 `0.01`，`foreach=False`。
- zero-grad 和 optimizer step 不套用社区 runner 原有的禁用编译策略。
- 不收集输出时返回 `loss.item()`；收集输出时保留预测结果。
- 禁用 CUDA matmul 和 cuDNN 的 TF32。

训练迭代包含输入克隆、清梯度、前向、loss、反向和 optimizer step。
不要用只测前向的 CUDA 脚本与 NPU 训练结果比较。
CUDA 编译路径可能同时包含 Triton kernel 和 CUDA 库 kernel，统计时均会计入。

## 5. NPU 对照命令

在已有 NPU 测试环境中，使用包含 device profiling 功能的 `test/cicd_benchmark` 分支，
并确认其 `benchmark` checkout 也固定为第 1 节的提交。
以下命令从 `pytorch_new/benchmarks/torchbench` 执行；设备可见性沿用该环境已确认的空闲卡配置。

```bash
set -o pipefail
npu_infer_dir="$PWD/results/dlrm_inference_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$npu_infer_dir"

python torchbench.py \
  --device npu --only dlrm \
  --performance --inference --backend inductor \
  --npu-backend triton --disable-aclgraph \
  --float32 --batch-size 2048 \
  --repeat 30 --iterations-per-run 1 \
  --enable-profiler 0 --prof-output-path "$npu_infer_dir/profile" \
  --device-perf --output result.csv --output-directory "$npu_infer_dir" \
  2>&1 | tee "$npu_infer_dir/run.log"
```

训练对照改为 `--training --batch-size 1024`，并另建训练结果目录。
NPU 命令不加 `--npu-parity`，该选项是 CUDA runner 提供的对齐开关。
CUDA 命令不加 `--npu-backend`，也不使用 `--disable-aclgraph`。

## 6. 采集轮次和耗时公式

两端都先完成原有的端到端性能测量，再分别额外执行 eager 和 compile 的 profile：

```text
每种模式：第 1～10 轮预热，第 11～20 轮正式采集
eager：10 轮预热 + 10 轮采集
compile：10 轮预热 + 10 轮采集
```

每轮调用一次模型迭代函数，然后同步设备。
这 10 轮采集不受 `--repeat`、`--iterations` 或 `--iterations-per-run` 控制。
整个程序还包含编译预热和端到端测量，因此总模型调用次数会超过 20 次，
不能据此判断 profile 超采。

NPU 读取本次 `kernel_details.csv` 的 `Duration(us)`；
CUDA 解析本次 `trace.json` 中 `ph="X"`、`cat="kernel"` 事件的 `dur`，单位同样为微秒。
CUDA 解析规则来自 `cuda_op_statistic.py`，不会生成中间的 `kernel_details.csv`。

```text
eager_device_latency   = eager 的 kernel 总耗时(us) / 10 / 1000
compile_device_latency = compile 的 kernel 总耗时(us) / 10 / 1000
device_speedup         = eager_device_latency / compile_device_latency
```

前两项单位均为 `ms/step`；加速比大于 1 表示 compile 更快。
例如 eager 为 `2 ms/step`、compile 为 `1 ms/step`，加速比就是 `2x`。

这是 kernel 累计耗时的单步均值，不是设备时间线的起止间隔。
多 stream 上重叠执行的 kernel 分别累加，不做重叠区间去重。
CUDA 不计 CPU 事件、CUDA runtime、memcpy 和 memset 事件。
端到端加速比仍使用计时结果的中位数，不能与这里的 device 指标混用。

## 7. 核对输出与归档结果

成功完成时，终端最终结果形式如下；数值仅为格式示例：

```text
[device] eager=2.000000 ms/step, compile=1.000000 ms/step, device_speedup=2.000x
```

开启 `--device-perf` 后，不在最终结果行打印端到端耗时和加速比。
`result.csv` 保留端到端列，并增加 device 列：

| 字段 | 含义 |
| --- | --- |
| `dev`、`name`、`batch_size` | 核对设备、模型和实际 batch size |
| `eager_device_latency` | eager 单步 kernel 累计耗时，毫秒 |
| `compile_device_latency` | compile 单步 kernel 累计耗时，毫秒 |
| `device_speedup` | eager device 耗时除以 compile device 耗时 |
| `device_profile_steps` | 应为 `10` |
| `speedup`、`abs_latency` | 原有端到端加速比、compile 端到端耗时，不能当作 device 结果 |

CUDA 结果目录如下：

```text
<结果目录>/
  run.log
  result.csv
  result_compilation_metrics.csv
  profile/dlrm/
    eager/performance_*/
      trace.json
      op_statistics.csv
    compile/performance_*/
      trace.json
      op_statistics.csv
```

每个 `op_statistics.csv` 与对应的 JSON 在同一目录，按设备和算子归并 GPU kernel。
优先通过 `External id` 关联 CPU 算子名，未关联时使用 kernel 名。
`Count` 是归入该行的 kernel 数量，`Total Time(us)` 是其累计设备耗时，
`Core Type` 为 `CUDA_CORE`；显示的 `Device_id` 从 1 开始，不代表物理卡编号。
CSV 展示值经过格式化，最终 device 加速比使用 JSON 中未经四舍五入的耗时。

如需独立复算，在 `benchmarks/dynamo` 执行以下命令。
推理使用 `"$cuda_infer_dir"`，训练则传入对应训练结果目录：

```bash
python - "$cuda_infer_dir" <<'PY'
import sys
from pathlib import Path
from device_profile import read_kernel_duration

root = Path(sys.argv[1]) / "profile" / "dlrm"
latencies = {}
for mode in ("eager", "compile"):
    traces = list((root / mode).glob("performance_*/trace.json"))
    assert len(traces) == 1, (mode, traces)
    latencies[mode] = read_kernel_duration(traces[0], 10)
    print(f"{mode}: {latencies[mode]:.6f} ms/step")
print(f"device_speedup: {latencies['eager'] / latencies['compile']:.3f}x")
PY
```

保留运行命令、日志、两端 runner 提交、benchmark 提交、依赖版本和设备型号，
将推理与训练结果分别比较。若日志出现编译失败、模型跳过或没有有效 kernel，
不能仅凭进程退出码或残留 CSV 判断成功，应检查本次最终结果及新生成的 profile。

## 8. 对齐范围与结果解释

`--npu-parity` 对齐上述训练和数值配置，不会把 NPU 专用模型补丁、
算子分解或 CPU fallback 移植到 CUDA。
如果 NPU 训练出现 `_embedding_bag_backward` 等 CPU fallback，应在结果中注明：
CPU 执行耗时不进入 NPU kernel duration，CUDA 若执行完整 GPU 路径，两端实际设备工作量存在差异。

两端的 `device_speedup` 分别表示各自设备上 compile 相对 eager 的收益。
例如 CUDA 为 `2x`、NPU 为 `1.1x`，只能说明各自相对 eager 的优化幅度，
不能单凭这个比值判断两种硬件谁的绝对性能更高。
对照时同时保留 eager、compile 的 `ms/step`，并注明硬件、版本和 fallback 情况。

实现参考：[采样与耗时计算](device_profile.py)、[CUDA 算子统计解析](cuda_op_statistic.py)、
[CUDA runner](torchbench.py) 和 [公共性能流程](common.py)。
