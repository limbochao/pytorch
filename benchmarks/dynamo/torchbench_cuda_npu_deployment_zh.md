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
| 端到端测量 | `--repeat 1 --iterations-per-run 1` | 相同 |
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
git log -1 --oneline

cd benchmarks/dynamo
git clone --depth=1 https://github.com/pytorch/benchmark.git benchmark
git -C benchmark fetch --depth=1 origin fb14629994956c16d27c7a9940c5f081afbf68dc
git -C benchmark checkout --detach fb14629994956c16d27c7a9940c5f081afbf68dc
git -C benchmark rev-parse HEAD

python -m pip install numpy pandas psutil pyyaml scipy tqdm
python benchmark/install.py dlrm
```

部署时无需重新获取 PyTorch tag 或检查分支共同祖先；使用上述测试分支即可。
benchmark 的 `rev-parse HEAD` 必须与固定提交一致。
模型依赖安装后，再执行上一节的包版本检查，并用 `python -m pip freeze` 保留依赖清单。

运行入口是当前目录下的 `torchbench.py`，不要换成 benchmark 仓库自己的模型运行入口。
runner 会加载这里的 `benchmark/torchbenchmark` 模型。
对于使用 TorchBench `DATA_PATH` 的模型，可设置数据集的绝对路径：

```bash
export TORCHBENCH_DATA_PATH=/absolute/path/to/torchbench_data
```

数据内容应与 NPU 相同，目录路径可以不同。此变量只指定数据位置，不负责下载数据；
使用模型自身固定数据路径的场景，还需按该模型要求准备数据。

### 3.1 加载阶段缺数据或依赖的修复

`python benchmark/install.py dlrm` 只安装 DLRM，不会准备全部 40 个模型的数据和依赖。
以下错误发生在模型加载阶段，尚未进入 Inductor 编译或 profile，不能据此判定模型不支持 CUDA。

| 模型 | 报错原因 | 修复内容 |
| --- | --- | --- |
| `functorch_maml_omniglot` | 找不到 `maml_omniglot/batch.pt` | 安装原始输入文件，目录属于 `maml_omniglot` |
| `pytorch_stargan` | 找不到 `list_attr_celeba.txt` | 解压完整的 `pytorch_stargan_inputs.tar.gz`，包含图片及标注 |
| `soft_actor_critic` | 找不到 `gymnasium` | 安装该模型 requirements 中的 `gymnasium`、`pygame`、`tensorboardX` |
| `speech_transformer` | 找不到 `train_chars.txt` | 解压完整的 `speech_transformer_inputs.tar.gz`，包含词表和 train/dev/test 输入 |
| `torch_multimodal_clip` | 找不到 `torchbenchmark.util.framework.fb` | 缺少模型本地 `.data` 目录，触发内部下载入口；补齐官方 `pizza.jpg` |

在 CUDA 的 `benchmarks/dynamo` 目录中，使用已固定版本的 benchmark 安装前四个模型：

```bash
python benchmark/install.py --verbose \
  functorch_maml_omniglot pytorch_stargan soft_actor_critic speech_transformer
```

如果安装时仍出现证书错误，注意 TIMM 的 `TORCHBENCH_TIMM_SKIP_SSL_VERIFY=1`
只作用于 TIMM runner；benchmark 安装器的 S3 下载使用独立的 `requests.get()`，仍会校验证书。
先用 `curl -k` 将三个原始输入文件放入安装器的缓存位置，再运行安装器。
以下命令使用同一组官方 URL，成功下载后才替换缓存文件；安装器发现文件已存在便跳过网络下载，
继续完成解压和依赖安装，无需修改 benchmark 源码或全局 Python 证书设置：

```bash
(
  set -e
  asset_base=https://ossci-datasets.s3.amazonaws.com/torchbench
  for asset in \
    models/maml_omniglot/batch.pt \
    data/pytorch_stargan_inputs.tar.gz \
    data/speech_transformer_inputs.tar.gz
  do
    target="benchmark/torchbenchmark/$asset"
    mkdir -p "$(dirname -- "$target")"
    curl -k --fail --location --retry 3 \
      "$asset_base/$asset" --output "$target.download"
    case "$asset" in
      *.tar.gz) tar -tzf "$target.download" >/dev/null ;;
    esac
    mv -- "$target.download" "$target"
  done
  python benchmark/install.py --verbose \
    functorch_maml_omniglot pytorch_stargan soft_actor_critic speech_transformer
)
```

如果错误来自 `pip` 下载依赖，而不是 S3 输入下载，可对这一次安装设置 `PIP_TRUSTED_HOST`。
使用官方 PyPI 时，在上面最后一条 `python benchmark/install.py` 前加以下环境变量：

```bash
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-} pypi.org pypi.python.org files.pythonhosted.org" \
  python benchmark/install.py --verbose \
  functorch_maml_omniglot pytorch_stargan soft_actor_critic speech_transformer
```

使用自定义 pip 镜像时，还需将其实际域名加入 `PIP_TRUSTED_HOST`。
此变量会传给安装器启动的 pip 子进程，但不会关闭 S3 的 requests 证书校验，
因此不能替代上面的输入文件预下载。保留当前环境代理；本次修复不要求绕过代理。

安装器将数据下载至 benchmark 源码树；它不读取 `TORCHBENCH_DATA_PATH`。
如果之前设置了自定义路径，应把完整的 `pytorch_stargan_inputs` 和 `speech_transformer_inputs`
目录放入该路径，或在本次运行中改用安装器的默认数据目录：

```bash
export TORCHBENCH_DATA_PATH="$PWD/benchmark/torchbenchmark/data/.data"
```

对于附件中已能导入 `torchmultimodal` 的 CLIP 环境，只补图片，保留与 NPU 对齐的已有依赖版本。
以下 URL 与固定版本模型的公开 `install.py` 一致；`-k` 沿用测试容器跳过证书校验的设置，
随后校验图片 SHA-256：

```bash
mkdir -p benchmark/torchbenchmark/models/torch_multimodal_clip/.data
curl -k --fail --location --retry 3 \
  https://ossci-assets.s3.us-east-1.amazonaws.com/2880px-Pizza-3007395.jpg \
  --output benchmark/torchbenchmark/models/torch_multimodal_clip/.data/pizza.jpg
python - <<'PY'
import hashlib
from pathlib import Path

path = Path("benchmark/torchbenchmark/models/torch_multimodal_clip/.data/pizza.jpg")
assert hashlib.sha256(path.read_bytes()).hexdigest() == (
    "580d3bb6381b0a249f9d90fb5e8b206c4a481308db78c493a74fd6c93984e0a0"
), f"Unexpected image contents: {path}"
print("CLIP image OK")
PY
```

也可以从已经跑通的 NPU 环境复制相同数据，避免再次联网下载。
NPU runner 将 CLIP 图片放在 `TORCHBENCH_DATA_PATH/pizza.jpg`；
CUDA 固定版本模型使用 `benchmark/torchbenchmark/models/torch_multimodal_clip/.data/pizza.jpg`，
需要复制到后者，单独设置 `TORCHBENCH_DATA_PATH` 不会改变 CLIP 的图片路径。
`batch.pt` 同样使用模型源码树中的固定位置，不受该变量影响。

安装器下载输入使用 `ossci-datasets.s3.amazonaws.com`，CLIP 图片来自 `ossci-assets.s3.us-east-1.amazonaws.com`。
修改 `HF_ENDPOINT` 不会改变这些下载地址；若代理或证书导致下载失败，可从 NPU 环境复制完整数据。
不要用空文件或随机输入代替缺失的原始数据。

重跑前检查关键文件，并打开图片确认有效：

```bash
python - <<'PY'
import os
from pathlib import Path
from PIL import Image
import gymnasium
import pygame
import tensorboardX

root = Path("benchmark/torchbenchmark")
data = Path(os.environ.get("TORCHBENCH_DATA_PATH", root / "data/.data"))
paths = [
    root / "models/maml_omniglot/batch.pt",
    data / "pytorch_stargan_inputs/data/celeba/list_attr_celeba.txt",
    data / "speech_transformer_inputs/input_data/lang_1char/train_chars.txt",
]
paths.extend(
    data / "speech_transformer_inputs/input_data" / split / name
    for split in ("train", "dev", "test") for name in ("data.json", "feats.1.ark")
)
for path in paths:
    assert path.is_file() and path.stat().st_size > 0, f"Missing or empty: {path}"
assert any((data / "pytorch_stargan_inputs/data/celeba/images").glob("*.jpg"))
with Image.open(root / "models/torch_multimodal_clip/.data/pizza.jpg") as image:
    image.verify()
print("Required inputs and dependencies found")
PY
```

随后使用**原来的命令、相同的 `--models` 和相同的输出目录**重新执行一键脚本。
已成功及已标记不支持的条目保留，只重试失败条目，成功后覆盖对应 CSV 汇总行。
旧版脚本把加载失败占位 CSV 中的 `batch_size=0` 报为 batch 不匹配；
新版先识别缺少 device profile 结果的占位行，并提示查看 `run.log` 中的原始异常。
有效性能结果仍会检查实际 batch 是否与 NPU 默认值一致，无需为上述加载错误修改 batch。

## 4. CUDA 测试命令

### 4.0 一键执行指定的 40 个模型

完成前面的环境部署，并准备好清单中各模型的依赖、权重及数据集后，在 CUDA PyTorch 仓库根目录执行：

```bash
bash benchmarks/dynamo/run_selected_device_performance.sh \
  --device 0 --output-dir ./cuda_device_results
```

仅执行部分模型时，使用 `--models`，多个模型名以空格分隔，可以跨测试套选择：

```bash
bash benchmarks/dynamo/run_selected_device_performance.sh \
  --device 0 --output-dir ./cuda_device_subset \
  --models BertForMaskedLM beit_base_patch16_224 dlrm nanogpt
```

模型名必须与 40 个模型清单中的名称完全一致；无效名称会在启动 CUDA 检查和创建结果目录前报错。
重复名称只执行一次。筛选后仍按原清单排序，每个模型先推理、后训练，
`summary.csv` 只包含选中模型的场景，`order` 从 1 开始连续编号。
省略 `--models` 时执行全部 40 个模型。

`--device` 指定一张物理 GPU，也支持 GPU UUID。省略时使用已有的 `CUDA_VISIBLE_DEVICES`，
未设置则使用 `0`；不接受多卡列表。脚本将选定的卡映射为进程内的 `cuda:0`，串行执行，
自动切换到 `benchmarks/dynamo`，因此可从其他目录通过脚本的绝对路径启动。
默认使用当前环境的 `python`，也可以通过 `PYTHON=/path/to/python` 指定解释器。
脚本不安装模型依赖；第 3 节的 `install.py dlrm` 只准备 DLRM，并不足以运行全部模型。

TIMM 的 6 个模型运行时默认使用 `HF_ENDPOINT=https://hf-mirror.com`，
如果调用者已设置非空的 `HF_ENDPOINT`，则保留该地址，不再强制覆盖。
脚本同时设置 `TORCHBENCH_TIMM_SKIP_SSL_VERIFY=1`，适配当前执行环境的证书要求。
后者让 TIMM runner 的 Hugging Face Hub 和 PyTorch URL 权重下载跳过证书校验，
同时禁用 Xet 和旧版 `hf_transfer`，让下载走已配置的 Python HTTP 客户端。
这些设置会写入该场景的 `command.sh`，仅传给 TIMM 进程；其他测试套保留调用者的环境配置。
单独执行 `timm_models.py` 时，可先在同一终端设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export TORCHBENCH_TIMM_SKIP_SSL_VERIFY=1
```

未设置 `TORCHBENCH_TIMM_SKIP_SSL_VERIFY=1` 时，单模型 runner 保持默认的证书校验。

如果通过当前代理访问镜像返回到 `https://huggingface.co/...` 的绝对地址 308 重定向，
Hub 可能在尚未取得 `X-Repo-Commit` 时就停止元数据查询并报错。
已确认官方站及 CDN 可达时，可直接指定官方入口：

```bash
export HF_ENDPOINT=https://huggingface.co
```

随后重新执行原来的一键命令，保持相同的模型选择和输出目录，即可只重试失败或未完成的场景。
仅切换下载入口时可复用原结果目录；实际模型及性能测试配置应保持一致。
保留已有可用代理；直连存在 DNS 解析失败时，禁用代理无法解决此问题。
每次实际使用的 endpoint 都会记录在对应的 `command.sh` 中。

执行顺序与 [40 个模型清单](npu_performance_batch_sizes_zh.md) 一致，
每个模型先推理、后训练；全量执行保留 **80 行**，筛选 N 个模型时保留 **2 × N 行**。
模型名单与 batch 默认值共同读取自 `npu_batch_sizes.py`，命令统一使用：

```text
--performance --repeat 1 --iterations 1 --iterations-per-run 1
--inference 或 --training
--float32 --device cuda --backend inductor --disable-cudagraphs --npu-parity
--enable-profiler 0 --device-perf
```

每个模型、每种模式分别保存日志、原始结果 CSV、可复现命令以及 profile：

```text
cuda_device_results/
  summary.csv
  perf/
    AlbertForMaskedLM_Infer/
      AlbertForMaskedLM_Infer_Eager_GPU.csv
      AlbertForMaskedLM_Infer_Inductor_GPU.csv
      AlbertForMaskedLM_Infer_Inductor_GPU_graph_001_model__0_inference_0.0_output_code.py
      AlbertForMaskedLM_Infer_Inductor_GPU_graph_001_model__0_inference_0.0_fx_graph_readable.py
    AlbertForMaskedLM_Train/
      ...
  runs/
    01_huggingface_AlbertForMaskedLM_inference/
      command.sh
      run.log
      result.csv
      debug/torch_compile_debug/run_*/torchinductor/
        model__0_inference_0.0/{output_code.py,fx_graph_readable.py,...}
      cache/{inductor,triton}/
      repair/                 # 自动补齐产物的独立命令、日志、debug 和缓存
      profile/AlbertForMaskedLM/
        eager/performance_*/{trace.json,op_statistics.csv}
        compile/performance_*/{trace.json,op_statistics.csv}
    02_huggingface_AlbertForMaskedLM_training/
      ...
```

`--output-dir` 相对于启动命令所在目录。首次运行可指定新目录或空目录，
省略时在启动目录创建带时间戳的结果目录。

每个实际执行的场景结束后，脚本自动把产物收集到 `输出目录/perf/模型名_Train或Infer/`：

- eager 的 `op_statistics.csv` 复制为 `场景名_Eager_GPU.csv`。
- Inductor 的 `op_statistics.csv` 复制为 `场景名_Inductor_GPU.csv`。
- 每个编译图的 `output_code.py`、`fx_graph_readable.py` 分别复制，保留原文件内容。
  文件名前缀为 `场景名_Inductor_GPU_graph_编号_原图目录名_`。

目录和 CSV 名称沿用参考 `perf` 目录的规则。单个图共 4 个文件；多个图分别保存两个 py，
包括训练的前向、反向图及 graph break 后的图，不会合并或只取其中一个。
编号按本次 debug 图目录的排序生成，前向/反向信息保留在原图目录名中。
原始 profile、debug 文件仍保留在对应的 `runs/` 场景目录。

脚本对每个 runner 进程自动设置以下环境变量，并写入该场景的 `command.sh`：

```text
TORCH_COMPILE_DEBUG=1
TORCH_COMPILE_DEBUG_DIR=<该场景 runs 目录>/debug
TORCHINDUCTOR_CACHE_DIR=<该场景 runs 目录>/cache/inductor
TRITON_CACHE_DIR=<该场景 runs 目录>/cache/triton
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
```

每个场景使用独立缓存目录，实际重试前清理该场景的 debug 和缓存，同时禁用编译图缓存复用，
使已有的本地或远程图缓存不会阻止本次生成 debug 产物。调用者原来的缓存目录不会被清理。
首次编译发生在预热阶段，device 耗时仍按预热后的 10 轮 profile 计算。

失败场景收集已经生成的部分文件，并以 `[artifacts] missing:` 提示缺失项；
加载失败或尚未生成编译图时，无法收齐全部文件。收集错误会单独打印，不覆盖性能测试状态，
`summary.csv` 继续表达模型性能测试是否成功。`unsupported` 场景不启动 runner，也不生成这些产物。

**失败后再次执行原命令，并保持同一个 `--output-dir`，即可自动重试，无需额外参数。**
脚本读取已有的 `summary.csv`，按模型及推理/训练模式分别判断：

- `ok`：保留已有耗时、日志和 profile，跳过性能测试；检查汇总产物，缺少时自动补齐。
- `unsupported`：保留不支持记录，跳过执行。
- `failed` 或 `pending`：重新执行；中断前尚未记录的场景也会补跑。

例如推理已经成功、训练失败，再次启动只会重试该模型的训练。
每次启动对待重试的场景各执行一次；仍失败则保留 `failed`，下次原命令可继续重试。
重试会清理该场景的旧 `result*` 文件、profile、debug 和缓存，并覆盖日志及运行命令；
同时清理该场景此前汇总的 GPU 文件，再按本次执行结果收集，防止残留旧图。
同目录下另行放入的 NPU 对照文件保留。成功后原位更新耗时、加速比、退出码和状态，
其他已成功场景的耗时、日志、profile 及已汇总文件保持不变。
全部场景已成功或不支持、且产物可从现有文件补齐时，不启动模型或 CUDA 检查。

**旧成功场景无需更换输出目录。再次执行原命令即可自动补齐编译产物。**

脚本先从原始 profile 和 debug 复制缺失文件；如果编译产物确实不存在，会使用独立的
`runs/场景目录/repair/`，调用 runner 的 `--performance --backend inductor --compile-only`。
此模式按原场景的推理/训练、FP32、编译后端以及 **summary 中保存的实际 batch** 加载模型，
执行与原编译预热一致的 5 轮，生成全部编译图后退出，不运行 eager 计时、性能比较或 profiler。
它不会将新的性能数值写回原 `summary.csv`、`result.csv`，也不覆盖原日志、profile 和已收集的 CSV。
例如旧结果是训练 batch 128，即使后来默认改成 16，补编译仍使用保存的 128。

若汇总 CSV 和它对应的原始 profile 文件也都丢失，则在同一个 `repair/` 目录执行独立性能采集，
仅补齐缺失的 CSV；原性能数值及其他已有 CSV 仍保留。这种补采 CSV 来自新采集，
不能将其中的 kernel 耗时当作原 summary 那次测量的明细。

补齐命令和输出分别记录在 `repair/command.sh`、`repair/run.log`；编译仍使用全新的独立缓存。
补齐成功前会核对保存的 batch 和全部图文件，失败的补编译产物不会作为已完成结果使用。
补齐失败时原性能状态仍为 `ok`，脚本返回 1；下次同一命令会继续尝试补齐。
全部产物齐全后，再次启动直接复用，不重新编译或测试。

重试时保持原来的模型选择、设备和测试配置；更换这些配置时使用新的输出目录。
已有非空目录必须包含格式正确且模型顺序匹配的 `summary.csv`，避免误用其他测试产物。
summary 在启动时列出本次计划的全部场景，未完成的状态为 `pending`；
每完成一个场景就通过临时文件替换来刷新对应行，保留原始顺序，不会追加重复行。
因此 Ctrl+C 中断后仍可复用同一命令继续执行。

`summary.csv` 字段包括：

| 字段 | 含义 |
| --- | --- |
| `order`、`suite`、`model`、`mode` | 原始顺序、测试套、模型名及推理/训练模式 |
| `batch_size` | 成功运行实际使用的 batch size，核对 NPU 默认值 |
| `eager_device_ms_per_step` | eager 单步 kernel 累计耗时，毫秒 |
| `inductor_device_ms_per_step` | Inductor 单步 kernel 累计耗时，毫秒 |
| `device_speedup` | eager device 耗时 / Inductor device 耗时 |
| `profile_steps` | 每种模式实际采集轮数，应为 10 |
| `status`、`exit_code`、`message` | `pending`、`ok`、`failed` 或 `unsupported`，以及退出码和原因 |
| `log_path`、`profile_path` | 该场景的日志和 profile 绝对路径 |

汇总只读取原始 CSV 的 `eager_device_latency`、`compile_device_latency`、`device_speedup`，
不使用端到端列；采样仍为每种模式预热 10 轮、采集 10 轮。
`llama` 和 `cm3leon_generate` 的训练标记为 `unsupported`，不启动 runner，耗时留空。
其他场景失败时标记为 `failed` 并继续后续模型，耗时留空，不能当作 0 ms 或有效加速比。
即使 runner 返回 0，缺少有效 device 结果、采样轮数或 batch 不匹配也会标记失败。
全部结束后如有性能测试失败或产物补齐失败，脚本返回 1；性能与产物均完成后返回 0。

### 单模型命令

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
  --repeat 1 --iterations-per-run 1 \
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
  --repeat 1 --iterations-per-run 1 \
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
  --repeat 1 --iterations-per-run 1 \
  --enable-profiler 0 --prof-output-path "$npu_infer_dir/profile" \
  --device-perf --output result.csv --output-directory "$npu_infer_dir" \
  2>&1 | tee "$npu_infer_dir/run.log"
```

训练对照改为 `--training --batch-size 1024`，并另建训练结果目录。
NPU 命令不加 `--npu-parity`，该选项是 CUDA runner 提供的对齐开关。
CUDA 命令不加 `--npu-backend`，也不使用 `--disable-aclgraph`。

## 6. 采集轮次和耗时公式

两端都先完成原有的端到端性能测量，再分别额外执行 eager 和 compile 的 profile：

文档命令使用 `--repeat 1 --iterations-per-run 1`，端到端部分只做一组 eager/compile 计时。
不能直接省略 `--repeat`，否则 runner 会使用默认的 30 组计时；profile 的 10 轮采集独立控制。

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
