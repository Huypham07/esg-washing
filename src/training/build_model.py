"""Factory tạo (model, tokenizer) cho train_model — tách riêng để cô lập code LLM.

- encoder: PhoBERT/BERT seq-classification (hành vi GIỮ NGUYÊN như train_model cũ).
- decoder_lora: LLM decoder (Qwen/Vistral/...) 4-bit QLoRA + classification-head.
  Dùng CHUNG WeightedTrainer/eval/metrics với encoder (loss qua .logits) -> so apples-to-apples.

Bake-off Phase 02b. Xem .claude/rules/code-style.md.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def build_model(config: dict, labels: list) -> tuple:
    """-> (model, tokenizer) theo config['model']['type'] ∈ {encoder, decoder_lora}."""
    model_cfg = config["model"]
    model_name = model_cfg["name"]
    model_type = model_cfg.get("type", "encoder")  # backward-compat: thiếu -> encoder (PhoBERT)
    num_labels = len(labels)
    id2label = {i: name for i, name in enumerate(labels)}
    label2id = {name: i for i, name in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_type == "decoder_lora":
        return _build_decoder_lora(model_cfg, model_name, num_labels, id2label, label2id, tokenizer)

    if model_type != "encoder":
        raise ValueError(f"Unknown model.type '{model_type}'. Available: ['encoder', 'decoder_lora']")

    # --- encoder (PhoBERT) — GIỮ NGUYÊN hành vi train_model cũ ---
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model, tokenizer


def _build_decoder_lora(
    model_cfg: dict, model_name: str, num_labels: int,
    id2label: dict, label2id: dict, tokenizer,
) -> tuple:
    """LLM decoder + 4-bit NF4 + LoRA classification-head (QLoRA)."""
    try:
        from transformers import BitsAndBytesConfig
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
    except ImportError as e:  # peft/bitsandbytes chưa cài
        raise ImportError(
            "decoder_lora cần `peft` + `bitsandbytes` (pip install -r requirements.txt). "
            f"Chi tiết: {e}"
        ) from e

    # Decoder (Qwen/Llama/Mistral) thường KHÔNG có pad_token -> dùng eos.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    # 4-bit phải nạp thẳng lên GPU (CPU không hỗ trợ) -> {"": 0} khi có CUDA.
    device_map = {"": 0} if torch.cuda.is_available() else None

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        quantization_config=bnb_config,
        device_map=device_map,
        id2label=id2label,
        label2id=label2id,
    )
    # seq-cls của decoder cần pad_token_id để xác định token cuối mỗi câu.
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False  # tránh cảnh báo khi gradient-checkpointing

    model = prepare_model_for_kbit_training(model)

    lora = model_cfg.get("lora", {})
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora.get("r", 16),
        lora_alpha=lora.get("alpha", 32),
        lora_dropout=lora.get("dropout", 0.05),
        bias="none",
        target_modules=lora.get("target_modules", "all-linear"),
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()  # cần cho gradient-checkpointing + PEFT
    return model, tokenizer
