"""
Benchmark script for evaluating inference performance of quantized LLaMA 2 (65B) and Mixtral 8x7B models.

This script runs two separate benchmarks:
  1. LLaMA 2 65B quantized with GPTQ on 2 GPUs.
  2. Mixtral 8x7B (MoE) quantized using 4-bit `BitsAndBytes` on 2 GPUs.

Each benchmark performs the following:
  - Loads the model and tokenizer from a specified local or remote path.
  - Runs a single prompt through the model using `generate()`.
  - Measures total inference time, tokens generated, and throughput (tokens/sec).
  - Prints diagnostic information including memory footprint, dtype, and device mapping.

Requirements:
  - PyTorch with multi-GPU support
  - `transformers`, `auto-gptq`, and `bitsandbytes`
  - Properly quantized model files in the expected paths

Example:
  $ python test_infer_Llama2_and_Mixtral.py

Note:
  - This script is intended for research or hardware benchmarking purposes.
  - Ensure that the models are downloaded and accessible at the specified paths.

Author(s):
    - Shang-Jui (Ray) Kuo
    - Adebayo Braimah (documentation)
    
Date: 2025
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from auto_gptq import AutoGPTQForCausalLM # Kept for Llama 2 if you run it

# Common settings
MAX_NEW_TOKENS = 100

# --- LLaMA 2 Function (flexible GPU count) ---
def benchmark_llama2_65b_gptq_ngpu(num_gpus_expected=2):
    print(f"\n=== Benchmark: LLaMA 2 65B q4_k_m ({num_gpus_expected}x GPU) ===")
    model_name_or_path = "./models/Llama2_65B_GPTQ" # Ensure this path is correct

    if not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus_expected:
        print(f"CUDA not available or less than {num_gpus_expected} GPUs found. Skipping Llama 2 benchmark.")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        model = AutoGPTQForCausalLM.from_quantized( # Make sure you have this import if running
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        print(f"Llama 2 65B model loaded with device_map='auto'.")
        param_dtype = next(model.parameters()).dtype
        print(f"Llama 2 65B model parameter dtype: {param_dtype}")
        print(f"Model memory footprint: {model.get_memory_footprint()} bytes")

        prompt = "What are the key ideas of quantum mechanics?"
        input_ids_tensor = tokenizer(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)

        start_time = time.time()
        output_ids = model.generate(
            input_ids=input_ids_tensor,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )
        end_time = time.time()
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
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
        print(f"LLaMA 2 65B GPTQ ({num_gpus_expected}x GPU) failed: {e}")
        import traceback
        traceback.print_exc()

# --- Mixtral Single Prompt Function (flexible GPU count, uses SDPA) ---
def benchmark_mixtral_single_prompt_ngpu(num_gpus_expected=2):
    print(f"\n=== Benchmark: Mixtral 8x7B (4-bit Quantized, Single Prompt, {num_gpus_expected}x GPU, SDPA) ===")
    model_name_or_path = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    if not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus_expected:
        print(f"CUDA not available or less than {num_gpus_expected} GPUs found. Skipping Mixtral single prompt benchmark.")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        if tokenizer.pad_token is None:
            print("Tokenizer does not have a pad token, setting it to eos_token.")
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
            attn_implementation="sdpa"  # Using SDPA for broader compatibility
        )
        print(f"Mixtral 8x7B model successfully loaded with device_map='auto'.")
        param_dtype = next(model.parameters()).dtype
        print(f"Mixtral 8x7B model parameter dtype: {param_dtype}")
        print(f"Model memory footprint: {model.get_memory_footprint()} bytes")

        messages = [
            {"role": "user", "content": "Explain the theory of relativity in simple terms."}
        ]
        # For apply_chat_template, it returns a dict if tokenized, or string if not.
        # If return_tensors="pt", it directly gives tensors.
        inputs_dict = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        input_ids_tensor = inputs_dict["input_ids"].to(next(model.parameters()).device)
        # Attention mask is part of inputs_dict if needed, but for single unpadded prompt, less critical for generate
        # attention_mask_tensor = inputs_dict.get("attention_mask").to(next(model.parameters()).device)


        print(f"Running single prompt inference on Mixtral...")
        start_time = time.time()
        output_ids = model.generate(
            input_ids=input_ids_tensor,
            # attention_mask=attention_mask_tensor, # Pass if generated and relevant
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        end_time = time.time()

        full_output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        tokens_in_prompt = input_ids_tensor.shape[1]
        tokens_generated = output_ids.shape[1] - tokens_in_prompt
        inference_time = end_time - start_time

        print(f"\nOutput:\n{full_output_text}")
        print(f"Inference time: {inference_time:.2f} seconds")
        if tokens_generated > 0:
            print(f"Tokens generated: {tokens_generated}")
            print(f"Time per token: {inference_time / tokens_generated:.4f} sec/token")
            print(f"Tokens per second: {tokens_generated / inference_time:.2f} tokens/sec")
        else:
            print(f"Tokens generated: {tokens_generated} (No new tokens were generated)")

    except Exception as e:
        print(f"Mixtral 8x7B (single prompt, {num_gpus_expected}x GPU, SDPA) failed: {e}")
        import traceback
        traceback.print_exc()

# --- Mixtral Batched Inference Function (flexible GPU count, uses SDPA) ---
def benchmark_mixtral_batched_ngpu(batch_size=4, num_gpus_expected=2):
    print(f"\n=== Benchmark: Mixtral 8x7B (4-bit Quantized, Batch Size: {batch_size}, {num_gpus_expected}x GPU, SDPA) ===")
    model_name_or_path = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    if not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus_expected:
        print(f"CUDA not available or less than {num_gpus_expected} GPUs found. Skipping Mixtral batched benchmark.")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        if tokenizer.pad_token is None:
            print("Tokenizer does not have a pad token, setting it to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
        # For decoder-only models, left padding is often preferred during training/batched generation
        tokenizer.padding_side = "left"

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
            attn_implementation="sdpa"  # Using SDPA
        )
        print(f"Mixtral 8x7B model successfully loaded with device_map='auto'.")
        param_dtype = next(model.parameters()).dtype
        print(f"Mixtral 8x7B model parameter dtype: {param_dtype}")
        print(f"Model memory footprint: {model.get_memory_footprint()} bytes")

        prompts_content = [
            "What are the strategic goals in the opening of a chess game?",
            "Write a short story about a robot who discovers music.",
            "Explain the concept of black holes to a child.",
            "List three benefits of regular exercise."
        ] * (batch_size // 4 + (1 if batch_size % 4 > 0 else 0))
        prompts_content = prompts_content[:batch_size]
        batched_messages = [[{"role": "user", "content": p}] for p in prompts_content]
        prompt_strings = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
        
        target_device = next(model.parameters()).device
        inputs = tokenizer(
            prompt_strings,
            return_tensors="pt",
            padding=True, # Pad to longest sequence in the batch
            truncation=True,
            max_length=tokenizer.model_max_length # Use model's max length for truncation
        ).to(target_device)

        print(f"Running batched inference (batch size: {batch_size}) on Mixtral...")
        start_time = time.time()
        output_ids_batch = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"], # Crucial for batched input with padding
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        end_time = time.time()

        total_tokens_generated_in_batch = 0
        decoded_outputs = []
        for i in range(output_ids_batch.shape[0]): # Iterate over batch
            # The input prompt part of the output is effectively inputs["input_ids"][i]
            # We need to count tokens generated *after* the prompt for each item.
            # output_ids_batch contains prompt + generated text
            
            # Find the length of the actual input prompt (excluding padding)
            input_prompt_length = torch.sum(inputs["attention_mask"][i]).item()
            
            # Decode the full sequence for this item
            full_output_text = tokenizer.decode(output_ids_batch[i], skip_special_tokens=True)
            decoded_outputs.append(full_output_text)

            # Calculate generated tokens for this sequence
            # Count non-pad tokens in the output sequence starting AFTER the prompt part
            generated_tokens_count_this_sequence = 0
            # Start counting from the end of the actual prompt in the output sequence
            # output_ids_batch might be longer due to padding if other sequences were longer
            for token_idx in range(input_prompt_length, output_ids_batch.shape[1]):
                if output_ids_batch[i, token_idx] != tokenizer.pad_token_id:
                    if output_ids_batch[i, token_idx] == tokenizer.eos_token_id:
                        break # Stop at EOS
                    generated_tokens_count_this_sequence += 1
                # If we encounter a pad token after where the prompt should have ended,
                # it means this sequence might have finished early with EOS then got padded.
                # However, for decoder-only models with left-padding, the generated content directly follows the prompt.
            total_tokens_generated_in_batch += generated_tokens_count_this_sequence


        inference_time = end_time - start_time
        print(f"\n--- Batch Output (first output shown) ---")
        if decoded_outputs:
            print(decoded_outputs[0])
        print(f"--- End of Batch Output ---")

        print(f"Batch inference time: {inference_time:.2f} seconds for {batch_size} prompts")
        if total_tokens_generated_in_batch > 0:
            print(f"Total tokens generated in batch: {total_tokens_generated_in_batch}")
            print(f"Average time per prompt: {inference_time / batch_size:.4f} sec/prompt")
            print(f"Overall tokens per second (throughput): {total_tokens_generated_in_batch / inference_time:.2f} tokens/sec")
        else:
            print(f"Total tokens generated in batch: {total_tokens_generated_in_batch} (Possible issue in token counting or no new tokens generated)")

    except Exception as e:
        print(f"Mixtral 8x7B (batched, {num_gpus_expected}x GPU, SDPA) failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    num_gpus = 0
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"CUDA available. Found {num_gpus} GPU(s).")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        torch.cuda.empty_cache()
    else:
        print("CUDA not available. Benchmarks requiring GPU will be skipped.")

    # Determine how many GPUs to tell the benchmarks to expect
    # This allows you to run on 1, 2, or 4 GPUs if available
    # Forcing it to 2 for V100 scenario, or 4 for RTX 8000.
    # Let's assume user will run on all available GPUs or a specific number.
    # If you have 4 GPUs (e.g. RTX 8000s) and want to use all:
    gpus_to_use = num_gpus # or set to 2 or 4 manually if you want to test specific counts

    # --- LLaMA 2 Benchmark (Optional) ---
    # Requires AutoGPTQForCausalLM and the model files
    # from auto_gptq import AutoGPTQForCausalLM # Ensure this is imported if running
    # if gpus_to_use > 0:
    #     print("\nRunning LLaMA 2 benchmark...")
    #     benchmark_llama2_65b_gptq_ngpu(num_gpus_expected=gpus_to_use)
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    # else:
    #     print("Skipping LLaMA 2 benchmark as no GPUs are configured for use.")

    # --- Mixtral Benchmarks ---
    if gpus_to_use > 0:
        print("\nRunning Mixtral benchmarks...")
        benchmark_mixtral_single_prompt_ngpu(num_gpus_expected=gpus_to_use)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # You can adjust the batch_size and gpus_to_use for the batched benchmark
        benchmark_mixtral_batched_ngpu(batch_size=4, num_gpus_expected=gpus_to_use)
        benchmark_mixtral_batched_ngpu(batch_size=8, num_gpus_expected=gpus_to_use)
    else:
        print("Skipping Mixtral benchmarks as no GPUs are configured for use.")

    print("\nAll benchmarks complete.")