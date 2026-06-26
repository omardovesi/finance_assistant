"""
Export fine-tuned model to quantized GGUF for Ollama — no llama.cpp build needed.
Run:  python export_gguf.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["UNSLOTH_FUSED_CE_COMPILE_DISABLE"] = "1"
from unsloth import FastLanguageModel

LORA_DIR  = "./finance-model-lora"
GGUF_PATH = "./finance-assistant-q4"   # Unsloth appends .gguf automatically

if not os.path.exists(LORA_DIR):
    raise FileNotFoundError(f"LoRA directory not found: {LORA_DIR}\nRun finetune.py first.")

print(f"Loading LoRA model from {LORA_DIR}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = LORA_DIR,
    max_seq_length = 512,
    dtype          = None,
    load_in_4bit   = True,
)

print(f"\nExporting to GGUF (Q4_K_M) — merges LoRA and quantizes in one step...")
print("This takes 5–15 minutes and uses significant RAM.\n")
model.save_pretrained_gguf(GGUF_PATH, tokenizer, quantization_method="q4_k_m")

gguf_file = GGUF_PATH + ".gguf" if not GGUF_PATH.endswith(".gguf") else GGUF_PATH
print(f"\n✅ GGUF saved to {os.path.abspath(gguf_file)}")
