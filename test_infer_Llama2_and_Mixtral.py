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

        # For AutoGPTQ, device_map="auto" handles multi-GPU.
        # Explicitly set torch_dtype to torch.float16 for V100s and GPTQ models.
        model = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path,
            device_map="auto",  # Key for multi-GPU
            torch_dtype=torch.float16, # Explicitly set to torch.float16
            trust_remote_code=True,
            # Regarding 'use_exllama':
            # AutoGPTQ will attempt to use the best available kernel.
            # For V100s, ExLlama (v1) might be an option if your AutoGPTQ version
            # has kernels compiled for it. ExLlamaV2 is generally for newer architectures.
            # You can explicitly try:
            # use_exllama=True, # or False to disable if causing issues or for comparison
            # disable_exllamav2=True, # Explicitly disable v2 if it's trying and failing on V100
        )
        print(f"Llama 2 65B model loaded on: {model.device}") # Should show 'meta' device or similar for device_map="auto"
        # It's good to check the actual dtype of the model parameters after loading
        if hasattr(model, 'model'): # Common for AutoGPTQ models
             print(f"Llama 2 65B model parameter dtype: {next(model.model.parameters()).dtype}")
        else:
             print(f"Llama 2 65B model parameter dtype: {next(model.parameters()).dtype}")
        print(f"Model memory footprint: {model.get_memory_footprint()} bytes")


        prompt = "What are the key ideas of quantum mechanics?"
        # Ensure input_ids are on the correct device, especially with device_map
        # For device_map="auto", the model might expect inputs on a specific GPU or CPU initially
        # model.device refers to the device of the *overall model object*, not necessarily where inputs should go first.
        # next(model.parameters()).device gives the device of the first set of parameters, a safer bet.
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)

        start_time = time.time()
        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False # Greedy decoding for consistent speed benchmark
        )
        end_time = time.time()

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        tokens_generated = output_ids.shape[1] - input_ids.shape[1]
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
    # You can use a local path if downloaded, or the Hugging Face model ID
    model_name_or_path = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # model_name_or_path = "./models/Mixtral" # If you have it downloaded

    # Check if CUDA is available and you have at least 2 GPUs
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("CUDA not available or less than 2 GPUs found. Skipping Mixtral 8x7B benchmark.")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token # Important for some models/batching

        # Configure 4-bit quantization for BitsAndBytes
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16, # V100s are good with float16
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4" # nf4 is a common choice
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            torch_dtype=torch.float16, # Match compute_dtype for consistency
            device_map="auto",          # Key for multi-GPU with accelerate
            trust_remote_code=True,
            attn_implementation="sdpa"  # Use Scaled Dot Product Attention (PyTorch 2.0+)
                                        # If "sdpa" causes issues, remove it or try "eager"
        )
        print(f"Mixtral 8x7B model loaded on: {model.device}") # Should show 'meta' device for device_map="auto"
        print(f"Mixtral 8x7B model parameter dtype: {next(model.parameters()).dtype}")
        print(f"Model memory footprint: {model.get_memory_footprint()} bytes")


        # For Mixtral Instruct, use the chat template
        messages = [
            {"role": "user", "content": "What are the strategic goals in the opening of a chess game?"}
        ]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(next(model.parameters()).device)

        start_time = time.time()
        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id, # Important if using batching or padding
            do_sample=True, # Mixtral Instruct often performs better with sampling
            temperature=0.6,
            top_p=0.9,
        )
        end_time = time.time()

        full_output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        output_text = full_output_text # For printing the full conversation context

        tokens_generated = output_ids.shape[1] - input_ids.shape[1]
        inference_time = end_time - start_time

        print(f"\nOutput:\n{output_text}") # full_output_text contains prompt + answer
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
    # Ensure you are logged in to Hugging Face if needed for private models or new Gemma etc.
    # from huggingface_hub import login
    # login("YOUR_HF_TOKEN") # If needed

    # It's good to clear CUDA cache if you run multiple benchmarks sequentially
    # especially if one fails due to OOM.
    if torch.cuda.is_available():
        print(f"CUDA available. Found {torch.cuda.device_count()} GPU(s).")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        torch.cuda.empty_cache()


    benchmark_llama2_65b_gptq_2gpu() # Uncomment to run

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    benchmark_mixtral_8x7b_4bit_2gpu()