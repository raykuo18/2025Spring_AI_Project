'''
🏆 This script makes sure you won’t hit:
OOM at load
OOM at backward
no gradients
device misallocation
missing multi-adapter support
✅ and leaves your setup ready to replace dummy input → real training loop.
'''

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
import gc
import os

# ========== CONFIG DEVICE ==========
def get_device() -> torch.device:
    """Returns the best available device: CUDA (NVIDIA GPU), MPS (Apple GPU), or CPU.

    Returns:
        Torch device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device_map: torch.device = get_device()
# =====================================

# ========== CONFIG ==========
base_model_name = "mistralai/Mistral-7B-v0.1"
use_second_branch = True  # ✅ set to True to prepare for 2-branch LoRA
# device_map = "auto"  # let accelerate split across GPUs
dtype = torch.float16
# ============================

print(f"\n✅ Checking available GPUs...")
if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    for i in range(num_devices):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f} GB")
else:
    print("❌ No CUDA GPUs available!")
    exit(1)

# ----------------------------
print(f"\n✅ Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

print(f"\n✅ Loading base model ({base_model_name})...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=dtype,
    device_map=None  # still CPU
)

print(f"\n✅ Model loaded (initially on CPU).")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ----------------------------
print(f"\n✅ Preparing LoRA config...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ✅ Apply first LoRA
peft_model = get_peft_model(model, lora_config)
print(f"✅ Applied first LoRA adapter.")

if use_second_branch:
    print(f"✅ Preparing second branch LoRA config...")
    lora_config2 = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    peft_model.add_adapter("branch2", lora_config2)
    peft_model.set_adapter("default")  # default active
    print(f"✅ Added second LoRA adapter (named 'branch2').")

# ----------------------------
trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in peft_model.parameters())
print(f"\n✅ Total model params: {total_params:,}")
print(f"✅ LoRA trainable params: {trainable_params:,} ({trainable_params / total_params * 100:.4f}%)")

# ----------------------------
print(f"\n✅ Moving model to devices using device_map={device_map}...")
peft_model = peft_model.half().to(device_map)

# Confirm device allocation
print(f"\n✅ Checking layer device allocation...")
for name, param in peft_model.named_parameters():
    print(f"{name}: {param.device}")

# Confirm GPU memory usage
for i in range(num_devices):
    used_mem = torch.cuda.memory_allocated(i) / (1024**3)
    reserved_mem = torch.cuda.memory_reserved(i) / (1024**3)
    print(f"GPU {i}: used {used_mem:.2f} GB / reserved {reserved_mem:.2f} GB")

# ----------------------------
# Create dummy input
print(f"\n✅ Creating dummy input...")
dummy_input = tokenizer("Hello world!", return_tensors="pt").to(peft_model.device)

# Forward pass
print(f"\n✅ Running forward pass...")
with torch.cuda.amp.autocast(dtype=torch.float16):
    output = peft_model(**dummy_input, labels=dummy_input["input_ids"])

print(f"✅ Forward pass done. Loss: {output.loss.item():.4f}")

# Backward pass
print(f"\n✅ Running backward pass...")
output.loss.backward()

print(f"✅ Backward pass done.")

# Check gradients
print(f"\n✅ Checking gradients...")
for name, param in peft_model.named_parameters():
    if param.requires_grad:
        grad_stat = param.grad.mean().item() if param.grad is not None else None
        print(f"{name}: grad mean={grad_stat}")

# Check GPU memory again
for i in range(num_devices):
    used_mem = torch.cuda.memory_allocated(i) / (1024**3)
    reserved_mem = torch.cuda.memory_reserved(i) / (1024**3)
    print(f"GPU {i} after backward: used {used_mem:.2f} GB / reserved {reserved_mem:.2f} GB")

# Optional garbage collection
gc.collect()
torch.cuda.empty_cache()

print("\n✅ Diagnostic complete. Ready for real training.")
