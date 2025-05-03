# === generate_outputs.py ===

import os
import pandas as pd
from tqdm import tqdm
import torch
from ctransformers import AutoModelForCausalLM

# Local model paths
local_model_paths = {
    "capybarahermes": "llm-models/capybarahermes-2.5-mistral-7b.Q4_K_M.gguf",
    "openchat": "llm-models/openchat_3.5.Q4_K_M.gguf",
}

# Simple FEN positions to test
fen_positions = [
    "r1bqkbnr/pppppppp/n7/8/8/N7/PPPPPPPP/R1BQKBNR w KQkq - 0 1",
    "rnbqkb1r/pp1ppppp/8/2p5/8/5NP1/PPPPPP1P/RNBQKB1R w KQkq - 0 2",
    "r1bq1rk1/ppp1bppp/2n2n2/3pp3/8/1PN1PN2/PBP2PPP/R2QKB1R w KQ - 0 7",
]

# Helper: auto-detect best gpu_layers
def select_gpu_layers():
    if torch.backends.mps.is_available():
        print("[Info] MPS (Apple Metal) detected. Using M1/M2 GPU.")
        return 20
    elif torch.cuda.is_available():
        print("[Info] CUDA GPU detected. Using NVIDIA GPU.")
        return 40
    else:
        print("[Info] No GPU detected. Using CPU.")
        return 0

gpu_layers = select_gpu_layers()

# Load local models
local_models = {}
for model_name, path in local_model_paths.items():
    print(f"Loading {model_name} from {path}...")
    local_models[model_name] = AutoModelForCausalLM.from_pretrained(
        path,
        model_type="mistral",
        gpu_layers=gpu_layers,
    )

# Define prompt
def generate_prompt(fen):
    return f"Given the chess board position (FEN): {fen}\nPlease recommend the next move and explain your reasoning clearly."

# Generate outputs
outputs = []

for fen in tqdm(fen_positions, desc="Generating outputs"):
    prompt = generate_prompt(fen)

    for model_name, model in local_models.items():
        pred = model(prompt, max_new_tokens=256)
        outputs.append({
            "fen": fen,
            "model": model_name,
            "output": pred,
        })

# Save to CSV
df = pd.DataFrame(outputs)
df.to_csv("outputs_generated.csv", index=False)

print("✅ Output saved to outputs_generated.csv")
