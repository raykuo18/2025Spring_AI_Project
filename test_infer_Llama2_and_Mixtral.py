import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM

def benchmark_llama2():
    print("\n=== Benchmark: LLaMA 2 65B q4_k_m ===")
    model_path = "./models/llama-2-65b-gptq"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )

    prompt = "What are the key ideas of quantum mechanics?"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    start = time.time()
    output_ids = model.generate(input_ids, max_new_tokens=100)
    end = time.time()

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    tokens_generated = output_ids.shape[1] - input_ids.shape[1]

    print(f"\nOutput: {output_text}")
    print(f"Inference time: {end - start:.2f} seconds")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Time per token: {(end - start)/tokens_generated:.4f} sec/token")

def benchmark_mixtral():
    print("\n=== Benchmark: Mixtral 8x7B ===")
    model_path = "./models/mixtral-8x7b-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )

    prompt = "What are the strategic goals in the opening of a chess game?"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    start = time.time()
    output_ids = model.generate(input_ids, max_new_tokens=100)
    end = time.time()

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    tokens_generated = output_ids.shape[1] - input_ids.shape[1]

    print(f"\nOutput: {output_text}")
    print(f"Inference time: {end - start:.2f} seconds")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Time per token: {(end - start)/tokens_generated:.4f} sec/token")

if __name__ == "__main__":
    benchmark_llama2()
    benchmark_mixtral()
