import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Configuration ---
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# ▼▼▼ IMPORTANT: REPLACE THIS WITH THE ACTUAL PATH TO YOUR LORA ADAPTER DIRECTORY ▼▼▼
LORA_ADAPTER_PATH = "./path/to/your/phase1_final_lora_adapter"
# For example, if your adapter is in './training_output/tinyllama_phase1/my_run_name/final_lora_adapter'
# then LORA_ADAPTER_PATH = "./training_output/tinyllama_phase1/my_run_name/final_lora_adapter"
# This directory should contain 'adapter_config.json' and 'adapter_model.safetensors' (or .bin)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def main():
    # 1. Load the base model and tokenizer
    print(f"Loading base model: {BASE_MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,  # Use float16 for faster inference if supported
        device_map="auto"           # Automatically map to GPU if available
    )
    print("Base model loaded.")

    print(f"Loading tokenizer for {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded.")

    # 2. Load the LoRA adapter and apply it to the base model
    # This creates a PeftModel which wraps your base model and applies the LoRA modifications.
    print(f"Loading LoRA adapter from: {LORA_ADAPTER_PATH}...")
    try:
        model_with_lora = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
        # No need to explicitly move to device if base_model used device_map="auto"
        # and the PeftModel wrapper respects it.
        print("LoRA adapter loaded and applied to the base model.")
    except Exception as e:
        print(f"Error loading LoRA adapter: {e}")
        print(f"Please ensure '{LORA_ADAPTER_PATH}' is a valid directory containing 'adapter_config.json' and 'adapter_model.safetensors' (or .bin).")
        return

    # Set the model to evaluation mode for inference
    model_with_lora.eval()

    # 3. Example Usage: Generate text
    # This prompt is similar to the structure your Phase 1 data might have used.
    # Replace with a prompt relevant to the tasks your LoRA adapter was trained on.
    example_prompt = "[INST] Predict the next move. [SEP] [FEN] rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1 [SEP] [PGN] 1. e4 [/INST]"
    
    print(f"\nGenerating text for prompt: \"{example_prompt}\"")

    inputs = tokenizer(example_prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    # Ensure inputs are on the same device as the model (model_with_lora.device could also be used)

    with torch.no_grad(): # Disable gradient calculations for inference
        try:
            # Generate text
            outputs = model_with_lora.generate(
                **inputs,
                max_new_tokens=20,  # Adjust as needed
                do_sample=True,     # Use sampling for more varied outputs
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id # Important if pad_token was set to eos_token
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # The generated_text will include your prompt.
            # To get only the newly generated part:
            prompt_length = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
            newly_generated_text = generated_text[prompt_length:].strip()

            print("\n--- Full Output (including prompt) ---")
            print(generated_text)
            print("\n--- Newly Generated Text ---")
            print(newly_generated_text)

        except Exception as e:
            print(f"Error during text generation: {e}")

    print("\nSimple LoRA adapter usage example complete.")

if __name__ == "__main__":
    main()
