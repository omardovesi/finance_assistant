Finance Assistant

A local, privacy-first personal finance assistant built by fine-tuning Mistral-7B-Instruct-v0.2 on a custom dataset and serving it through Ollama. Everything runs on your own machine, so your financial questions never leave your device.

What it does

The assistant is tuned to give clear, practical guidance on core personal finance topics:

Budgeting: building and sticking to a monthly budget
Emergency funds: how much to save and where to keep it
Debt payoff: strategies for paying down balances
Tax-advantaged accounts: 401(k), IRA, Roth, HSA and how they fit together

Disclaimer: This project is for educational purposes only. It is not a licensed financial advisor, and its output is not financial, tax, or legal advice. Verify important decisions with a qualified professional.

How it was built
Component	Details
Base model	Mistral-7B-Instruct-v0.2
Method	LoRA (r=16) on a 4-bit quantized base (QLoRA), via Unsloth
Training framework	TRL (SFTTrainer / SFTConfig)
Dataset	~450 hand-built instruction/response examples (finance_data_fixed.jsonl)
Training environment	Google Colab
Export format	GGUF, q4_k_m quantization (~4.4 GB)
Serving	Ollama

Training configuration

Epochs: 2
Per-device batch size: 1
Gradient accumulation steps: 8
Max sequence length: 1024
Project structure
finance-assistant/
├── Modelfile                  # Ollama config: base GGUF + system prompt
├── finetune.py                # Fine-tuning script
├── finance_data_fixed.jsonl   # Training dataset
└── finance-assistant-q4.gguf  # Quantized model (not tracked in git)
Getting started
Prerequisites
Ollama installed
~5 GB of free disk space
8 GB+ RAM recommended (a GPU speeds things up but isn't required)
1. Get the model file

Place finance-assistant-q4.gguf in the project folder next to the Modelfile.

2. Register the model with Ollama

The Modelfile already points at the GGUF and contains the finance system prompt:

FROM ./finance-assistant-q4.gguf

From the project directory, run:

bash
ollama create finance-assistant -f Modelfile

Windows tip: run PowerShell as administrator to avoid file permission errors during model creation.

3. Chat with it
bash
ollama run finance-assistant

Example prompts:

How big should my emergency fund be if I have a variable income?
I have $8k in credit card debt at 22% APR. Avalanche or snowball?
Should I prioritize my 401(k) match or a Roth IRA first?
Reproducing the fine-tune

Fine-tuning was done on Google Colab.

Upload finance_data_fixed.jsonl and finetune.py to a Colab GPU runtime.
Install Unsloth and TRL.
Run the training script with the configuration above.
Export with model.save_pretrained_gguf() using q4_k_m quantization.
Copy the resulting GGUF to your local machine.
Troubleshooting notes
Local fine-tuning on RTX 50-series (Blackwell, sm_120) on Windows: the CUDA 12.8+ requirement conflicts with current PyTorch/Unsloth version constraints. Colab is the reliable workaround.
Out-of-memory errors on Colab: use batch size 1 and sequence length 1024.
Checkpoint PicklingError in TRL: use SFTConfig directly instead of TrainingArguments.
Drive mount failures during GGUF export: export to the local Colab disk first, then copy the finished file to Drive.
Limitations
Trained on a small dataset (~450 examples), so coverage is narrow and it can be wrong or overconfident.
Tuned for general U.S.-style personal finance; it does not know your personal situation or current tax law.
May hallucinate figures such as contribution limits or rates. Always double-check numbers.
Roadmap
 Expand the dataset across more finance categories
 Add an evaluation set and benchmark against the base model
 Wrap Ollama in a simple chat UI
