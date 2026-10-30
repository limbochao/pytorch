# 选定模型的 NPU performance batch 对照表

范围仅包含本次指定的 **40 个模型**：HuggingFace 18 个、TIMM 6 个、TorchBench 16 个。
CUDA 在 `--performance --npu-parity` 下采用下表默认值，不需要额外指定 `--batch-size`。
清单外模型、不启用 `--npu-parity` 的运行以及 accuracy 模式均保持社区默认 batch 选择。

配置集中在 [npu_batch_sizes.py](npu_batch_sizes.py)。
来源为 NPU runner 提交 `9fb9ca72e8cf`，以及 TorchBench 模型提交
`fb14629994956c16d27c7a9940c5f081afbf68dc`。
这是默认值同步，不是这 40 个模型均已完成 CUDA 训练、推理验证的声明。

## HuggingFace

默认值按 NPU 的模型列表、`BATCH_SIZE_DIVISORS` 以及强制 batch 规则计算。
其中 MT5 的有效默认值是 **4**，不是列表中 32 除以 2 得到的 16。

| 模型 | 训练 batch | 推理 batch |
| --- | ---: | ---: |
| AlbertForMaskedLM | 4 | 4 |
| BartForCausalLM | 4 | 4 |
| BertForMaskedLM | 16 | 16 |
| DistilBertForMaskedLM | 128 | 128 |
| DistillGPT2 | 16 | 16 |
| ElectraForCausalLM | 32 | 32 |
| GPT2ForSequenceClassification | 4 | 4 |
| LayoutLMForMaskedLM | 16 | 16 |
| MBartForCausalLM | 4 | 4 |
| MT5ForConditionalGeneration | 4 | 4 |
| MegatronBertForCausalLM | 4 | 4 |
| MobileBertForMaskedLM | 128 | 128 |
| OPTForCausalLM | 2 | 2 |
| PLBartForCausalLM | 8 | 8 |
| PegasusForCausalLM | 32 | 32 |
| RobertaForCausalLM | 16 | 16 |
| TrOCRForCausalLM | 32 | 32 |
| XGLMForCausalLM | 8 | 8 |

## TIMM

NPU 直接使用其 `timm_models_list.txt` 中的 batch，不套用 CUDA 社区的额外 divisor。

| 模型 | 训练 batch | 推理 batch |
| --- | ---: | ---: |
| beit_base_patch16_224 | 128 | 128 |
| deit_base_distilled_patch16_224 | 128 | 128 |
| deit_tiny_patch16_224.fb_in1k | 128 | 128 |
| nfnet_l0 | 128 | 128 |
| vit_base_patch14_dinov2.lvd142m | 128 | 128 |
| vit_base_patch16_siglip_256 | 128 | 128 |

## TorchBench

默认值包含模型类及其父类的默认设置、训练 batch 表、NPU 模型覆盖，以及
`npu_support.py` 中影响 batch 的补丁。

| 模型 | 训练 batch | 推理 batch |
| --- | ---: | ---: |
| BERT_pytorch | 16 | 16 |
| basic_gnn_edgecnn | 1 | 1 |
| basic_gnn_gcn | 1 | 1 |
| basic_gnn_sage | 1 | 1 |
| cm3leon_generate | 1，见下方说明 | 1 |
| dcgan | 64 | 64 |
| dlrm | 1024 | 2048 |
| functorch_maml_omniglot | 1 | 1 |
| lennard_jones | 4096 | 4096 |
| llama | 无默认值，见下方说明 | 32 |
| nanogpt | 1 | 1 |
| nvidia_deeprecommender | 256 | 256 |
| pytorch_stargan | 16 | 16 |
| soft_actor_critic | 32768 | 32768 |
| speech_transformer | 32 | 1 |
| torch_multimodal_clip | 32 | 32 |

训练场景有两项需要单独记录：

- `llama`：固定模型版本没有 `DEFAULT_TRAIN_BSIZE`。自动选训练 batch 时会报出无默认值，
  本次不为它虚构训练 batch，也不新增训练实现。
- `cm3leon_generate`：NPU runner 会将 `--training` 自动切换为推理。表中 1 是其请求时涉及的 batch，
  不表示训练性能有效；本次也不修改 CUDA 模型的训练能力。

`soft_actor_critic` 原模型默认是 256，但 NPU 补丁将其改为 32768。
因为它禁止自定义 batch，CUDA 会在构造该模型期间临时对齐类默认值，构造后恢复，
避免模型把 32768 当成非法自定义 batch。其余 NPU 算子补丁不在本次修改范围。

## 运行方式

在 `benchmarks/dynamo` 下执行，设备编号替换为可用 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 python huggingface.py \
  --only AlbertForMaskedLM --device cuda \
  --performance --training --backend inductor --repeat 1 \
  --float32 --npu-parity --disable-cudagraphs \
  --enable-profiler 0 --device-perf --prof-output-path "$PWD/profile_hf_train" \
  --output hf_train.csv --output-directory "$PWD"

CUDA_VISIBLE_DEVICES=0 python timm_models.py \
  --only beit_base_patch16_224 --device cuda \
  --performance --training --backend inductor --repeat 1 \
  --float32 --npu-parity --disable-cudagraphs \
  --enable-profiler 0 --device-perf --prof-output-path "$PWD/profile_timm_train" \
  --output timm_train.csv --output-directory "$PWD"

CUDA_VISIBLE_DEVICES=0 python torchbench.py \
  --only dlrm --device cuda \
  --performance --training --backend inductor --repeat 1 \
  --float32 --npu-parity --disable-cudagraphs \
  --enable-profiler 0 --device-perf --prof-output-path "$PWD/profile_tb_train" \
  --output tb_train.csv --output-directory "$PWD"
```

推理时将 `--training` 改为 `--inference`，并使用不同的输出文件和 profile 目录。
其他模型替换 `--only` 的名称，runner 仍按所属 suite 选择。
HF 和 TIMM 需要先安装与 NPU 测试一致的 transformers、timm 等模型依赖。
profile 仍为 eager、compile 各预热 10 轮、采集 10 轮。
命令中的 `--repeat 1` 只将端到端测量设为一组，不改变 profile 采样轮次。

允许自定义 batch 的模型仍可显式传入 `--batch-size` 或 batch-size-file 覆盖默认值。
不支持自定义 batch 的 TorchBench 模型保留原有限制。
NPU 的 MT5 及部分 TorchBench 模型会强制覆盖显式 batch，因此手动修改时应检查两端
CSV 的实际 `batch_size`，不能只核对命令。

此处对 HF/TIMM 同步的是默认 batch 和 TF32 开关；优化器、模型输入生成及训练实现
沿用各自社区 runner。此前的普通 SGD 训练迭代对齐仅应用于 TorchBench。
设备耗时计算、JSON 与 `op_statistics.csv` 产物说明见
[部署与对齐指南](torchbench_cuda_npu_deployment_zh.md)。
