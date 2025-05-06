"""Script for downloading a Hugging Face causal language model, applying a LoRA (Low-Rank Adaptation) adapter,
saving both the base and adapter models locally, and preparing for conversion to GGUF format.

This script supports three model architectures:
    - TinyLLaMA
    - Gemma-2B
    - Phi-2

Steps performed:
    1. Selects and loads the specified model and tokenizer from Hugging Face.
    2. Saves the base model and tokenizer locally in Hugging Face format.
    3. Applies a LoRA adapter to attention projection modules (`q_proj`, `v_proj`).
    4. Saves only the LoRA adapter weights for efficient parameter-efficient fine-tuning (PEFT).
    5. Prints trainable parameter count for inspection.
    6. Outputs shell and web instructions for converting to the GGUF format used by llama.cpp.

Attributes:
    model_name (str): Specifies which base model to use. Must be one of {"TinyLLaMA", "Gemma-2B", "Phi-2"}.
    HF_MODEL_NAME (str): Hugging Face model identifier based on `model_name`.
    TARGET_MODULES (List[str]): List of attention modules targeted by LoRA.
    HF_CACHE_DIR (str): Directory to cache Hugging Face models locally.
    BASE_MODEL_DIR (str): Path to store the full base model and tokenizer.
    LORA_ADAPTER_DIR (str): Path to store the LoRA adapter weights.

Requirements:
    - `transformers` library from Hugging Face
    - `peft` library for LoRA/PEFT support
    - Local write permissions to cache and output directories

Example:
    To use TinyLLaMA:
        Set `model_name = "TinyLLaMA"` at the top of the script and run.

Next Steps (printed at runtime):
    1. Convert base model to GGUF using `convert-hf-to-gguf.py` from `llama.cpp`.
    2. Convert LoRA adapter to GGUF using the Hugging Face web UI.
    3. Optionally quantize the GGUF model for efficient inference.

Author(s):
    - Shang-Jui (Ray) Kuo
    - Adebayo Braimah (documentation)
Date:
    May 2025
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
import os

# ✅ SELECT MODEL HERE
model_name = "TinyLLaMA"  # options: "TinyLLaMA", "Gemma-2B", "Phi-2"

if model_name == "TinyLLaMA":
    HF_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    TARGET_MODULES = ["q_proj", "v_proj"]  # llama-based attention
elif model_name == "Gemma-2B":
    HF_MODEL_NAME = "google/gemma-2b"
    TARGET_MODULES = ["q_proj", "v_proj"]  # gemma also uses llama-like attention
elif model_name == "Phi-2":
    HF_MODEL_NAME = "microsoft/phi-2"
    TARGET_MODULES = ["q_proj", "v_proj"]  # phi-2 uses same naming (check exact arch if issues)
else:
    raise ValueError(f"Unknown model name: {model_name}")

HF_CACHE_DIR = "./pretrained_cache"
BASE_MODEL_DIR = f"./base_model/{model_name}"
LORA_ADAPTER_DIR = f"./lora_adapter/{model_name}"

print(f"✅ Loading model: {HF_MODEL_NAME}")

# ------------------------------
# ✅ Step 1: Load base model → save Hugging Face format
model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_NAME,
    cache_dir=HF_CACHE_DIR,
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    HF_MODEL_NAME,
    cache_dir=HF_CACHE_DIR
)

# ✅ save base model
os.makedirs(BASE_MODEL_DIR, exist_ok=True)
model.save_pretrained(f"{BASE_MODEL_DIR}/model")
tokenizer.save_pretrained(f"{BASE_MODEL_DIR}/tokenizer")

# print(f"✅ Base model saved to {BASE_MODEL_DIR}")

# ------------------------------
# ✅ Step 2: Create LoRA adapter
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=TARGET_MODULES,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Optional: print trainable param count
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ LoRA trainable params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.4f}%)")

# Optional training placeholder
# from transformers import Trainer, TrainingArguments
# trainer = Trainer(model=model, args=..., train_dataset=...)
# trainer.train()

# ✅ save LoRA adapter
os.makedirs(LORA_ADAPTER_DIR, exist_ok=True)
model.save_pretrained(LORA_ADAPTER_DIR)

print(f"✅ LoRA adapter saved to {LORA_ADAPTER_DIR}")

# ------------------------------
print(f"""
✅ Next steps:

1️⃣ Convert base model → GGUF locally:
$ cd llama.cpp
$ python3 convert-hf-to-gguf.py --model-dir {BASE_MODEL_DIR}/model --outfile ./base_model_{model_name}.gguf

2️⃣ Convert LoRA adapter → GGUF (web app):
👉 https://huggingface.co/spaces/ggml-org/gguf-my-lora

3️⃣ Quantize:
$ ./quantize ./base_model_{model_name}.gguf ./base_model_{model_name}.q4_k_m.gguf q4_K_M
""")
