"""Selected performance batch defaults from the NPU runner at 9fb9ca72e8cf.

TorchBench model source: fb14629994956c16d27c7a9940c5f081afbf68dc.
Each pair is (training, inference); None means no NPU training default.
The DINOv2 training batch is overridden to 16 for this CUDA test.
"""


NPU_BATCH_SIZES = {
    "huggingface": {
        "AlbertForMaskedLM": (4, 4),
        "BartForCausalLM": (4, 4),
        "BertForMaskedLM": (16, 16),
        "DistilBertForMaskedLM": (128, 128),
        "DistillGPT2": (16, 16),
        "ElectraForCausalLM": (32, 32),
        "GPT2ForSequenceClassification": (4, 4),
        "LayoutLMForMaskedLM": (16, 16),
        "MBartForCausalLM": (4, 4),
        "MT5ForConditionalGeneration": (4, 4),
        "MegatronBertForCausalLM": (4, 4),
        "MobileBertForMaskedLM": (128, 128),
        "OPTForCausalLM": (2, 2),
        "PLBartForCausalLM": (8, 8),
        "PegasusForCausalLM": (32, 32),
        "RobertaForCausalLM": (16, 16),
        "TrOCRForCausalLM": (32, 32),
        "XGLMForCausalLM": (8, 8),
    },
    "timm_models": {
        "beit_base_patch16_224": (128, 128),
        "deit_base_distilled_patch16_224": (128, 128),
        "deit_tiny_patch16_224.fb_in1k": (128, 128),
        "nfnet_l0": (128, 128),
        "vit_base_patch14_dinov2.lvd142m": (16, 128),
        "vit_base_patch16_siglip_256": (128, 128),
    },
    "torchbench": {
        "BERT_pytorch": (16, 16),
        "basic_gnn_edgecnn": (1, 1),
        "basic_gnn_gcn": (1, 1),
        "basic_gnn_sage": (1, 1),
        "cm3leon_generate": (1, 1),
        "dcgan": (64, 64),
        "dlrm": (1024, 2048),
        "functorch_maml_omniglot": (1, 1),
        "lennard_jones": (4096, 4096),
        "llama": (None, 32),
        "nanogpt": (1, 1),
        "nvidia_deeprecommender": (256, 256),
        "pytorch_stargan": (16, 16),
        "soft_actor_critic": (32768, 32768),
        "speech_transformer": (32, 1),
        "torch_multimodal_clip": (32, 32),
    },
}


def get_npu_batch_size(args, suite, model_name, batch_size):
    if not args.npu_parity or not args.performance or batch_size is not None:
        return batch_size
    sizes = NPU_BATCH_SIZES.get(suite, {}).get(model_name)
    if sizes is None:
        return batch_size
    batch_size = sizes[0 if args.training else 1]
    if batch_size is None:
        raise NotImplementedError(f"{model_name}: the pinned NPU benchmark has no default training batch size")
    return batch_size
