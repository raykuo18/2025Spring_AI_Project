import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig

model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
gguf_file = "tinyllama-1.1b-chat-v1.0.Q6_K.gguf"

TARGET_MODULES = ["q_proj", "v_proj"]  # attention weights

# LORA_PATH: str = (
#     "/Users/adebayobraimah/Desktop/projects/2025Spring_AI_Project/src/models/final_lora_adapter"
# )

LORA_PATH: str = "/home/adbraimah/cse537/2025Spring_AI_Project/src/models/final_lora_adapter"


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


# ──────────────────────────────────────────────────────────────────────────────
# 1)  Load a 4-bit-quantised base model + tokenizer
# ──────────────────────────────────────────────────────────────────────────────
# def load_base(dtype=torch.float16):
#     """
#     Returns: (model, tokenizer) — both ready for inference on CPU/GPU.
#     """
#     qcfg = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_compute_dtype=dtype,
#     )
#     # model = AutoModelForCausalLM.from_pretrained(
#     #     BASE_MODEL,
#     #     quantization_config=qcfg,
#     #     device_map="auto",
#     #     trust_remote_code=True,
#     # )

#     # # Load GGUF model (will dequantize into standard PyTorch model)
#     # model = AutoModelForCausalLM.from_pretrained(
#     #     model_id,
#     #     gguf_file=BASE_MODEL,  # this triggers GGUF parsing
#     # )

#     tok = AutoTokenizer.from_pretrained(
#         model_id,
#         gguf_file=gguf_file,
#         quantization_config=qcfg,
#         device_map=get_device(),
#         trust_remote_code=True,
#     )
#     model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=gguf_file)

#     # tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
#     tok.pad_token = tok.eos_token  # ensure no pad-token issues
#     return model.eval(), tok


def load_base(dtype=torch.float16, lora_path: str = LORA_PATH):
    """
    Loads the base GGUF model and (optionally) injects LoRA weights.

    Args:
        dtype: Desired torch dtype (default: float16)
        lora_path: Optional path to LoRA adapter folder

    Returns:
        model, tokenizer
    """
    tok = AutoTokenizer.from_pretrained(
        model_id,
        gguf_file=gguf_file,
        trust_remote_code=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        gguf_file=gguf_file,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).eval()

    if lora_path:
        # Apply LoRA adapter on top of the base model
        print(f"🔁 Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(base_model, lora_path).eval()
    else:
        model = base_model

    tok.pad_token = tok.eos_token
    return model, tok


# ──────────────────────────────────────────────────────────────────────────────
# 2)  Attach a fresh (random) LoRA adapter and register it under *name*
# ──────────────────────────────────────────────────────────────────────────────
def add_blank_lora(model, name: str, r: int = 8, lora_alpha: int = 32):
    """
    Adds a new LoRA adapter with random weights and returns its *name*.
    Useful for quick α–β mixing demos when you haven’t trained the adapters yet.
    """
    cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    # get_peft_model returns a *new* model wrapper; we just need the adapter added:
    lora_model = get_peft_model(model, cfg)
    lora_model.base_model.model.add_adapter(name, cfg)
    return name


# ──────────────────────────────────────────────────────────────────────────────
# 3)  Greedy generation with α–β blended logits
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def gen(
    prompt: str,
    model,
    tok,
    tac_name: str,
    strat_name: str,
    alpha_w: float = 1.0,
    beta_w: float = 1.0,
    max_new_tokens: int = 128,
):
    """
    Blends the per-step logits of two adapters:
        logits = α · logits_tactical  +  β · logits_strategic
    and returns the decoded text.
    """
    # normalise for safety (so alpha+beta = 1 unless both zero)
    alpha_w = float(alpha_w)
    beta_w = float(beta_w)
    if alpha_w + beta_w == 0:
        alpha_w = beta_w = 1.0
    alpha = alpha_w / (alpha_w + beta_w)
    beta = 1.0 - alpha

    device = next(model.parameters()).device
    ids = tok(prompt, return_tensors="pt").to(device).input_ids

    for _ in range(max_new_tokens):
        # branch-1 (tactical)
        model.set_adapter(tac_name)
        logits_tac = model(ids).logits[:, -1, :]

        # branch-2 (strategic)
        model.set_adapter(strat_name)
        logits_str = model(ids).logits[:, -1, :]

        # weighted sum & greedy pick
        blended = alpha * logits_tac + beta * logits_str
        next_id = torch.argmax(blended, dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=-1)

        if next_id.item() == tok.eos_token_id:
            break

    return tok.decode(ids[0], skip_special_tokens=True)
