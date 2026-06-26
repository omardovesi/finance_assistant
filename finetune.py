import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Disable torch.compile on Unsloth's chunked CE loss: on Windows WDDM,
# Dynamo traces accumulate_chunk with symbolic shapes where the chunk-size
# dimension (s0) and the label dimension (s2) are unrelated symbols, causing
# a "batch_size mismatch" TorchRuntimeError that bypasses the except block.
# Setting this flag before import makes _FUSED_CE_COMPILE_SUPPORTED=False so
# accumulate_chunk always runs in eager mode (slower but correct on WDDM).
os.environ["UNSLOTH_FUSED_CE_COMPILE_DISABLE"] = "1"
from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# ─────────────────────────────────────────────
# 0. QUICK SANITY CHECK
# ─────────────────────────────────────────────
print("=" * 60)
print("FINANCE ASSISTANT FINE-TUNING SCRIPT")
print("=" * 60)

if not torch.cuda.is_available():
    raise SystemError(
        "\n\n❌ CUDA not available. Make sure PyTorch was installed with CUDA support.\n"
        "Run: pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
    )

print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
print(f"✅ VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ─────────────────────────────────────────────
# 1. SYSTEM PROMPT (your full prompt baked in)
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a personal finance assistant designed to help me manage my money, stay on track with financial goals, and take actionable steps toward building wealth. Your behavior must follow these principles:
- Be practical, structured, and step-by-step
- Give clear, actionable advice (not vague suggestions)
- Prioritize financial fundamentals over risky or advanced strategies
- Never assume unknown information — ask questions when needed
- Do not hallucinate financial rules — stick to widely accepted best practices

Your responsibilities:
1. Track and analyze my spending habits
2. Help me create and adjust budgets
3. Recommend financial goals (emergency fund, saving, investing)
4. Suggest next steps like opening a Roth IRA, brokerage account, or savings account
5. Guide me step-by-step on how to actually set these up
6. Keep me accountable and adjust advice based on my behavior

When giving advice, ALWAYS follow this priority order:
1. Build an emergency fund (3–6 months expenses)
2. Pay off high-interest debt
3. Take advantage of employer retirement match (if applicable)
4. Maximize tax-advantaged accounts (Roth IRA, 401k)
5. Invest in brokerage accounts after the above

When analyzing finances:
- Break down spending into categories
- Identify waste or inefficiencies
- Compare spending vs goals
- Give 2–3 specific improvements to implement immediately

When recommending something (like a Roth IRA), include:
- What it is (simple explanation)
- Why it matters for me specifically
- Whether I'm eligible (if relevant)
- EXACT steps to set it up (platforms, actions, order)

Tone: Clear, direct, slightly strict but supportive — like a disciplined coach who wants me to improve.

Output format:
- Summary
- Key Insights
- Action Plan (numbered steps)
- Next Questions (if you need more info)

Priority order for advice:
1. Emergency fund first
2. High-interest debt second
3. 401k employer match third
4. Roth IRA / 401k max fourth
5. Brokerage investing fifth

Your goal: help me build discipline, clarity, and long-term wealth."""


# ─────────────────────────────────────────────
# 2. LOAD AND FORMAT YOUR DATASET
# ─────────────────────────────────────────────
DATA_FILE = "finance_data_fixed.jsonl"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"\n\n❌ Could not find {DATA_FILE}\n"
        f"Make sure finance_data.jsonl is in: {os.getcwd()}\n"
    )

raw_data = []
with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            raw_data.append(obj)
        except json.JSONDecodeError as e:
            print(f"⚠️  Skipping line {line_num} (invalid JSON): {e}")

print(f"\n✅ Loaded {len(raw_data)} training examples from {DATA_FILE}")

# Format into Mistral instruction format (no system prompt in training —
# system prompt goes in the Ollama Modelfile at inference time, not in every sample)
def format_example(example):
    prompt = example.get("prompt", "")
    response = example.get("response", "")
    formatted = f"<s>[INST] {prompt} [/INST] {response} </s>"
    return {"text": formatted}

formatted_data = [format_example(ex) for ex in raw_data]
dataset = Dataset.from_list(formatted_data)

# Split: 90% train, 10% validation
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]

print(f"✅ Training examples:   {len(train_dataset)}")
print(f"✅ Validation examples: {len(eval_dataset)}")


# ─────────────────────────────────────────────
# 3. LOAD BASE MODEL WITH UNSLOTH
# ─────────────────────────────────────────────
print("\n📦 Loading base model (Mistral-7B)...")
print("   This downloads ~4 GB on first run — please wait...\n")

MAX_SEQ_LENGTH = 512   # Mistral-7B GQA (8 KV heads) keeps attention memory small; batch=1 fits in 8 GB at seq=512

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name   = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    max_seq_length = MAX_SEQ_LENGTH,
    dtype          = None,   # Auto-detect: float16 on most GPUs
    load_in_4bit   = True,   # QLoRA: fits in 6-10 GB VRAM
)

print("✅ Base model loaded successfully")


# ─────────────────────────────────────────────
# 4. APPLY LoRA ADAPTERS
# ─────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r              = 16,        # LoRA rank — higher = more capacity, more VRAM
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha     = 16,
    lora_dropout   = 0,
    bias           = "none",
    use_gradient_checkpointing = "unsloth",  # Saves VRAM
    random_state   = 42,
    use_rslora     = False,
    loftq_config   = None,
)

print("✅ LoRA adapters applied")


# ─────────────────────────────────────────────
# 4.5 FILTER SEQUENCES LONGER THAN MAX_SEQ_LENGTH
# ─────────────────────────────────────────────
# Unsloth's fused CE loss truncates hidden_states but NOT labels when a
# sequence exceeds max_seq_length, causing a shape mismatch at runtime.
# Pre-filter both splits so nothing over-length ever reaches the model.
print("\n🔍 Filtering examples that exceed max sequence length...")

def _fits_in_context(example):
    ids = tokenizer(example["text"], add_special_tokens=True)["input_ids"]
    return len(ids) <= MAX_SEQ_LENGTH

n_train_before = len(train_dataset)
n_eval_before  = len(eval_dataset)
train_dataset = train_dataset.filter(_fits_in_context)
eval_dataset  = eval_dataset.filter(_fits_in_context)
print(f"✅ Training:   {n_train_before} → {len(train_dataset)} (dropped {n_train_before - len(train_dataset)} long examples)")
print(f"✅ Validation: {n_eval_before} → {len(eval_dataset)} (dropped {n_eval_before - len(eval_dataset)} long examples)")


# ─────────────────────────────────────────────
# 5. TRAINING CONFIGURATION
# ─────────────────────────────────────────────
OUTPUT_DIR = "./finance-model-output"

training_args = SFTConfig(
    output_dir              = OUTPUT_DIR,
    num_train_epochs         = 3,
    per_device_train_batch_size  = 1,
    per_device_eval_batch_size   = 1,
    gradient_accumulation_steps  = 8,
    learning_rate           = 2e-4,
    fp16                    = False,
    bf16                    = True,
    logging_steps           = 10,
    eval_strategy           = "steps",
    eval_steps              = 50,
    save_strategy           = "steps",
    save_steps              = 50,
    save_total_limit        = 3,
    load_best_model_at_end  = True,
    warmup_steps            = 20,
    weight_decay            = 0.01,
    lr_scheduler_type       = "cosine",
    optim                   = "paged_adamw_8bit",
    seed                    = 42,
    report_to               = "none",
    dataloader_num_workers  = 0,
    max_seq_length          = MAX_SEQ_LENGTH,
    dataset_text_field      = "text",
    packing                 = False,
)


# ─────────────────────────────────────────────
# 6. INITIALIZE TRAINER
# ─────────────────────────────────────────────
trainer = SFTTrainer(
    model            = model,
    tokenizer        = tokenizer,
    train_dataset    = train_dataset,
    eval_dataset     = eval_dataset,
    args             = training_args,
)

print("\n✅ Trainer initialized")
print(f"✅ Output directory: {OUTPUT_DIR}")


# ─────────────────────────────────────────────
# 7. RUN TRAINING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("🚀 STARTING TRAINING")
print("=" * 60)
print("This will take 30–90 minutes depending on your GPU.")
print("Loss should decrease steadily — good target is below 1.0")
print("Do NOT close this window during training.\n")

torch.cuda.empty_cache()
trainer_stats = trainer.train()

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE")
print("=" * 60)
print(f"Total training time: {trainer_stats.metrics['train_runtime'] / 60:.1f} minutes")
print(f"Final training loss: {trainer_stats.metrics['train_loss']:.4f}")


# ─────────────────────────────────────────────
# 8. SAVE THE FINE-TUNED MODEL
# ─────────────────────────────────────────────
LORA_SAVE_DIR   = "./finance-model-lora"
MERGED_SAVE_DIR = "./finance-model-merged"

print(f"\n💾 Saving LoRA adapter weights to {LORA_SAVE_DIR}...")
model.save_pretrained(LORA_SAVE_DIR)
tokenizer.save_pretrained(LORA_SAVE_DIR)
print("✅ LoRA adapters saved")

print(f"\n💾 Merging LoRA into base model and saving to {MERGED_SAVE_DIR}...")
print("   This takes 5–10 minutes...\n")

# Use Unsloth's merge method — transformers 5.x save_pretrained() fails on 4-bit
# models due to unimplemented revert_weight_conversion; Unsloth handles it correctly.
model.save_pretrained_merged(MERGED_SAVE_DIR, tokenizer, save_method="merged_16bit")

print("✅ Merged model saved")


# ─────────────────────────────────────────────
# 9. SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("🎉 ALL DONE — NEXT STEPS")
print("=" * 60)

