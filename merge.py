"""
Merge LoRA adapters into the base model and save for GGUF conversion.
Run after finetune.py has completed:  python merge.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["UNSLOTH_FUSED_CE_COMPILE_DISABLE"] = "1"
from unsloth import FastLanguageModel

LORA_DIR   = "./finance-model-lora"
MERGED_DIR = "./finance-model-merged"

if not os.path.exists(LORA_DIR):
    raise FileNotFoundError(f"LoRA directory not found: {LORA_DIR}\nRun finetune.py first.")

print(f"Loading LoRA model from {LORA_DIR}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = LORA_DIR,
    max_seq_length = 512,
    dtype          = None,
    load_in_4bit   = True,
)

print(f"\nMerging and saving to {MERGED_DIR}  (5-10 min, uses significant RAM)...")
model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")

print(f"\n✅ Merged model saved to {os.path.abspath(MERGED_DIR)}")
