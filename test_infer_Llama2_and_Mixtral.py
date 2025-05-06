import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from auto_gptq import AutoGPTQForCausalLM # For Llama 2 - ensure this is correctly installed & built

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
        model = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        print(f"Llama 2 65B model loaded with device_map='auto'. First param on: {next(model.parameters()).device}")
        param_dtype = next(model.parameters()).dtype
        print(f"Llama 2 65B model parameter dtype: {param_dtype}")
        if hasattr(model, 'get_memory_footprint'):
            print(f"Model memory footprint: {model.get_memory_footprint()} bytes")

        prompt = "What are the key ideas of quantum mechanics?"
        input_ids_tensor = tokenizer(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)

        print(f"Llama 2 - Input IDs shape: {input_ids_tensor.shape}")

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
            attn_implementation="sdpa"
        )
        print(f"Mixtral 8x7B model successfully loaded with device_map='auto'. First param on: {next(model.parameters()).device}")
        param_dtype = next(model.parameters()).dtype
        print(f"Mixtral 8x7B model parameter dtype: {param_dtype}")
        if hasattr(model, 'get_memory_footprint'):
            print(f"Model memory footprint: {model.get_memory_footprint()} bytes")

        messages = [{"role": "user", "content": "Explain the theory of relativity in simple terms."}]
        
        inputs_dict = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True # Important for instruct models
        )
        target_device = next(model.parameters()).device
        input_ids_for_generate = inputs_dict["input_ids"].to(target_device)
        
        attention_mask_for_generate = inputs_dict.get("attention_mask")
        if attention_mask_for_generate is None:
            print("WARNING: tokenizer.apply_chat_template did not return an attention_mask for a single prompt. Creating a default one (all ones).")
            attention_mask_for_generate = torch.ones_like(input_ids_for_generate)
        
        attention_mask_for_generate = attention_mask_for_generate.to(target_device)

        print(f"Mixtral Single Prompt - Input IDs shape: {input_ids_for_generate.shape}")
        print(f"Mixtral Single Prompt - Attention Mask shape: {attention_mask_for_generate.shape}")

        print(f"Running single prompt inference on Mixtral...")
        start_time = time.time()
        output_ids = model.generate(
            input_ids=input_ids_for_generate, # Changed to keyword argument for consistency
            attention_mask=attention_mask_for_generate, 
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        end_time = time.time()

        full_output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        tokens_in_prompt = input_ids_for_generate.shape[1]
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
            attn_implementation="sdpa"
        )
        print(f"Mixtral 8x7B model successfully loaded with device_map='auto'. First param on: {next(model.parameters()).device}")
        param_dtype = next(model.parameters()).dtype
        print(f"Mixtral 8x7B model parameter dtype: {param_dtype}")
        if hasattr(model, 'get_memory_footprint'):
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
        
        practical_max_tokenization_length = 2048
        if hasattr(tokenizer, 'model_max_length') and isinstance(tokenizer.model_max_length, int) and tokenizer.model_max_length < 32769:
             practical_max_tokenization_length = min(practical_max_tokenization_length, tokenizer.model_max_length)
        else:
            practical_max_tokenization_length = min(practical_max_tokenization_length, 32768) 
        print(f"Using practical_max_tokenization_length for tokenizer: {practical_max_tokenization_length}")

        target_device = next(model.parameters()).device
        inputs = tokenizer(
            prompt_strings,
            return_tensors="pt",
            padding=True, 
            truncation=True,
            max_length=practical_max_tokenization_length
        ).to(target_device)

        print(f"Mixtral Batched - Input IDs shape: {inputs['input_ids'].shape}")
        print(f"Mixtral Batched - Attention Mask shape: {inputs['attention_mask'].shape}")
        print(f"Running batched inference (batch size: {batch_size}) on Mixtral...")
        start_time = time.time()
        output_ids_batch = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        end_time = time.time()

        total_tokens_generated_in_batch = 0
        decoded_outputs = []
        for i in range(output_ids_batch.shape[0]):
            input_prompt_length = torch.sum(inputs["attention_mask"][i]).item()
            full_output_text = tokenizer.decode(output_ids_batch[i], skip_special_tokens=True)
            decoded_outputs.append(full_output_text)
            
            generated_tokens_count_this_sequence = 0
            current_output_sequence = output_ids_batch[i]
            # Count tokens in the output *after* the actual prompt length
            # Note: input_prompt_length is the count of non-padded tokens in the input.
            # Generation starts after these tokens.
            # The output sequence includes the prompt.
            for token_idx in range(input_prompt_length, current_output_sequence.shape[0]):
                if current_output_sequence[token_idx] != tokenizer.pad_token_id:
                    if current_output_sequence[token_idx] == tokenizer.eos_token_id:
                        break 
                    generated_tokens_count_this_sequence += 1
                else: 
                    break 
            total_tokens_generated_in_batch += generated_tokens_count_this_sequence

        inference_time = end_time - start_time
        print(f"\n--- Batch Output (first output shown) ---")
        if decoded_outputs: print(decoded_outputs[0])
        print(f"--- End of Batch Output ---")

        print(f"Batch inference time: {inference_time:.2f} seconds for {batch_size} prompts")
        if total_tokens_generated_in_batch > 0 and batch_size > 0 :
            print(f"Total tokens generated in batch: {total_tokens_generated_in_batch}")
            print(f"Average time per prompt: {inference_time / batch_size:.4f} sec/prompt")
            print(f"Overall tokens per second (throughput): {total_tokens_generated_in_batch / inference_time:.2f} tokens/sec")
        else:
            print(f"Total tokens generated in batch: {total_tokens_generated_in_batch} (Possible issue in token counting or no new tokens generated for this batch size)")

    except Exception as e:
        print(f"Mixtral 8x7B (batched, {num_gpus_expected}x GPU, SDPA) failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # To silence the tokenizer parallelism warning, you can set this environment variable
    # either in your shell before running the script, or using os.environ in Python:
    # import os
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"

    num_gpus = 0
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"CUDA available. Found {num_gpus} GPU(s).")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        torch.cuda.empty_cache()
    else:
        print("CUDA not available. Benchmarks requiring GPU will be skipped.")

    gpus_to_use = num_gpus 
    if gpus_to_use == 0 :
        print("No GPUs available to use for benchmarks.")
    
    # --- LLaMA 2 Benchmark ---
    # if gpus_to_use > 0:
    #     print("\nRunning LLaMA 2 benchmark...")
    #     benchmark_llama2_65b_gptq_ngpu(num_gpus_expected=gpus_to_use) 
    #     if torch.cuda.is_available(): torch.cuda.empty_cache()
    # else:
    #     print("Skipping LLaMA 2 benchmark as no GPUs are configured for use.")

    # --- Mixtral Benchmarks ---
    if gpus_to_use > 0:
        print("\nRunning Mixtral benchmarks...")
        benchmark_mixtral_single_prompt_ngpu(num_gpus_expected=gpus_to_use)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        benchmark_mixtral_batched_ngpu(batch_size=4, num_gpus_expected=gpus_to_use)
        if torch.cuda.is_available():
             torch.cuda.empty_cache()
        # You can add more calls with different batch sizes:
        # benchmark_mixtral_batched_ngpu(batch_size=8, num_gpus_expected=gpus_to_use)
    else:
        print("Skipping Mixtral benchmarks as no GPUs are configured for use.")

    print("\nAll benchmarks complete.")