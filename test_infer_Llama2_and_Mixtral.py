import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from auto_gptq import AutoGPTQForCausalLM

# Common settings
MAX_NEW_TOKENS = 100

def benchmark_llama2_65b_gptq_2gpu():
    print("\n=== Benchmark: LLaMA 2 65B q4_k_m (2x V100) ===")
    model_name_or_path = "./models/Llama2_65B_GPTQ" # Ensure this path is correct

    # Check if CUDA is available and you have at least 2 GPUs
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("CUDA not available or less than 2 GPUs found. Skipping Llama 2 65B benchmark.")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        model = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        print(f"Llama 2 65B model loaded on: {model.device}")
        if hasattr(model, 'model'):
             print(f"Llama 2 65B model parameter dtype: {next(model.model.parameters()).dtype}")
        else:
             print(f"Llama 2 65B model parameter dtype: {next(model.parameters()).dtype}")
        print(f"Model memory footprint: {model.get_memory_footprint()} bytes")

        prompt = "What are the key ideas of quantum mechanics?"
        input_ids_tensor = tokenizer(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)

        start_time = time.time()
        # MODIFICATION HERE: Pass input_ids as a keyword argument
        output_ids = model.generate(
            input_ids=input_ids_tensor,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )
        end_time = time.time()

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Ensure correct calculation of generated tokens
        # input_ids_tensor might have a different shape than output_ids[0] if padding was involved on model side,
        # but for single sequence, this should be okay.
        tokens_generated = output_ids.shape[1] - input_ids_tensor.shape[1]
        inference_time = end_time - start_time

        print(f"\nOutput: {output_text}")
        print(f"Inference time: {inference_time:.2f} seconds")
        if tokens_generated > 0:
            print(f"Tokens generated: {tokens_generated}")
            print(f"Time per token: {inference_time / tokens_generated:.4f} sec/token")
            print(f"Tokens per second: {tokens_generated / inference_time:.2f} tokens/sec")
        else:
            print(f"Tokens generated: {tokens_generated} (No new tokens were generated)")

    except Exception as e:
        print(f"LLaMA 2 65B GPTQ (2x V100) failed: {e}")
        import traceback
        traceback.print_exc()


def benchmark_mixtral_8x7b_4bit_2gpu():
    print("\n=== Benchmark: Mixtral 8x7B (4-bit Quantized, 2x V100) ===")
    model_name_or_path = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # model_name_or_path = "./models/Mixtral" # If you have it downloaded

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("CUDA not available or less than 2 GPUs found. Skipping Mixtral 8x7B benchmark.")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa"
        )
        print(f"Mixtral 8x7B model loaded on: {model.device}")
        print(f"Mixtral 8x7B model parameter dtype: {next(model.parameters()).dtype}")
        print(f"Model memory footprint: {model.get_memory_footprint()} bytes")

        messages = [
            {"role": "user", "content": "What are the strategic goals in the opening of a chess game?"}
        ]
        # Renamed to input_ids_tensor for clarity, though not strictly necessary if scoped locally
        input_ids_tensor = tokenizer.apply_chat_template(messages, return_tensors="pt").to(next(model.parameters()).device)

        start_time = time.time()
        # The Mixtral part using AutoModelForCausalLM usually handles positional input_ids fine.
        # No change needed here unless it also errors.
        output_ids = model.generate(
            input_ids_tensor, # This is typically fine for standard HF models
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        end_time = time.time()

        full_output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        output_text = full_output_text

        tokens_generated = output_ids.shape[1] - input_ids_tensor.shape[1]
        inference_time = end_time - start_time

        print(f"\nOutput:\n{output_text}")
        print(f"Inference time: {inference_time:.2f} seconds")
        if tokens_generated > 0:
            print(f"Tokens generated: {tokens_generated}")
            print(f"Time per token: {inference_time / tokens_generated:.4f} sec/token")
            print(f"Tokens per second: {tokens_generated / inference_time:.2f} tokens/sec")
        else:
            print(f"Tokens generated: {tokens_generated} (No new tokens were generated)")

    except Exception as e:
        print(f"Mixtral 8x7B (4-bit, 2x V100) failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"CUDA available. Found {torch.cuda.device_count()} GPU(s).")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        torch.cuda.empty_cache()

    benchmark_llama2_65b_gptq_2gpu()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    benchmark_mixtral_8x7b_4bit_2gpu()