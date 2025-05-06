#!/usr/bin/env python3

import json
import re
import argparse
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from typing import List, Dict, Optional, Set, IO
import glob

# Common settings for generation
MAX_NEW_TOKENS_DEFAULT = 150

# Global model and tokenizer to load only once
mixtral_model = None
mixtral_tokenizer = None

def load_mixtral_model_and_tokenizer(model_name_or_path: str, num_gpus_expected: Optional[int] = None):
    global mixtral_model, mixtral_tokenizer
    if mixtral_model is not None and mixtral_tokenizer is not None:
        print("Mixtral model and tokenizer already loaded.")
        return

    print(f"Loading Mixtral model: {model_name_or_path}...")
    if num_gpus_expected is not None and (not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus_expected):
        print(f"Warning: CUDA not available or less than {num_gpus_expected} GPUs found. Attempting to load on available devices.")
    try:
        mixtral_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        if mixtral_tokenizer.pad_token is None:
            print("Tokenizer does not have a pad token, setting it to eos_token.")
            mixtral_tokenizer.pad_token = mixtral_tokenizer.eos_token
        mixtral_tokenizer.padding_side = "left"

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        mixtral_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa"
        )
        print(f"Mixtral 8x7B model successfully loaded. First param on: {next(mixtral_model.parameters()).device}")
    except Exception as e:
        print(f"Failed to load Mixtral model: {e}")
        import traceback; traceback.print_exc()
        mixtral_model = None; mixtral_tokenizer = None
        raise

def get_mixtral_explanations_batched(
    teacher_prompts: List[str],
    batch_size: int,
    max_new_tokens: int
) -> List[str]:
    global mixtral_model, mixtral_tokenizer
    if mixtral_model is None or mixtral_tokenizer is None:
        raise RuntimeError("Mixtral model and tokenizer not loaded.")

    all_explanations = []
    num_prompts = len(teacher_prompts)

    for i in tqdm(range(0, num_prompts, batch_size), desc="Generating Explanations with Mixtral"):
        batch_teacher_prompts = teacher_prompts[i:i + batch_size]
        
        practical_max_tokenization_length = getattr(mixtral_tokenizer, 'model_max_length', 4096)
        if practical_max_tokenization_length > 8192: practical_max_tokenization_length = 8192
        
        inputs = mixtral_tokenizer(
            batch_teacher_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=practical_max_tokenization_length - max_new_tokens
        ).to(mixtral_model.device)

        try:
            output_ids_batch = mixtral_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                pad_token_id=mixtral_tokenizer.pad_token_id,
                eos_token_id=mixtral_tokenizer.eos_token_id,
                do_sample=True, temperature=0.6, top_p=0.9,
            )
        except Exception as e_gen:
            print(f"Error during model.generate for batch starting {i}: {e_gen}")
            all_explanations.extend(["GENERATION_ERROR"] * len(batch_teacher_prompts))
            continue

        for j in range(output_ids_batch.shape[0]):
            prompt_len = inputs["attention_mask"][j].sum().item()
            generated_ids = output_ids_batch[j, prompt_len:]
            explanation = mixtral_tokenizer.decode(generated_ids, skip_special_tokens=True)
            all_explanations.append(explanation)
    return all_explanations

def post_process_explanation(text: str, original_prompt: Optional[str]=None) -> str:
    processed_text = text.strip()
    boilerplate_starts = [
        "Explanation:", "Okay, here's an explanation:", "Sure, I can explain that.", "Here is the explanation:",
        "Here's a concise explanation:", "The explanation is as follows:"
    ]
    for bp_start in boilerplate_starts:
        if processed_text.lower().startswith(bp_start.lower()):
            processed_text = processed_text[len(bp_start):].strip()
    if processed_text.startswith("[ASSISTANT]"):
        processed_text = processed_text[len("[ASSISTANT]"):].strip()
    
    # (Further post-processing based on observed outputs can be added here)

    sentences = re.split(r'(?<=[.!?])\s+', processed_text)
    max_sentences = 4 # Example: limit to max 4 sentences
    if len(sentences) > max_sentences:
        processed_text = " ".join(sentences[:max_sentences])
    if processed_text and processed_text[-1] not in ".!?":
        processed_text += "."
    processed_text = re.sub(r'\s\s+', ' ', processed_text).strip()
    processed_text = processed_text.replace("\n", " ").strip()
    return processed_text

def load_processed_task_ids(output_folder_path: str) -> Set[str]:
    processed_ids = set()
    if not os.path.isdir(output_folder_path): return processed_ids
    part_files = sorted(glob.glob(os.path.join(output_folder_path, "training_data_part_*.jsonl")))
    if not part_files: print(f"No existing part files found in {output_folder_path}."); return processed_ids
    print(f"Found existing output part files: {part_files}. Checking processed tasks...")
    for part_file in part_files:
        try:
            with open(part_file, "r", encoding="utf-8") as f:
                for line in f:
                    try: data = json.loads(line); processed_ids.add(data["task_id"])
                    except (json.JSONDecodeError, KeyError): print(f"Warn: Skip invalid line/no task_id in {part_file}: {line.strip()}")
        except Exception as e: print(f"Error reading {part_file}: {e}. Continuing...")
    print(f"Found {len(processed_ids)} already processed task_ids."); return processed_ids

def main():
    # <<< --- GLOBAL DECLARATION MOVED TO THE TOP --- >>>
    global mixtral_model, mixtral_tokenizer

    parser = argparse.ArgumentParser(
        description="Step 2: Run Mixtral inference, save to checkpointed files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input-prompts-file", required=True, help="Path to JSONL file from Step 1.")
    parser.add_argument("-o", "--output-training-folder", required=True, help="Path to FOLDER for final training data parts.")
    parser.add_argument("--model-name-or-path", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1", help="Mixtral model path.")
    parser.add_argument("--batch-size", type=int, default=4, help="Mixtral inference batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS_DEFAULT, help="Max new tokens for Mixtral.")
    parser.add_argument("--num-gpus", type=int, default=None, help="Num GPUs for model loading. Default: auto.")
    parser.add_argument("--num-samples", type=int, default=None, help="Max new prompts to process. Default: all new.")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Save new part file every N samples. 0 for single file.")
    args = parser.parse_args()

    if not os.path.exists(args.input_prompts_file): parser.error(f"Input file not found: {args.input_prompts_file}")
    if not os.path.exists(args.output_training_folder): os.makedirs(args.output_training_folder, exist_ok=True)
    elif not os.path.isdir(args.output_training_folder): parser.error(f"Output path not a directory: {args.output_training_folder}")

    try:
        # --- Load Mixtral Model and Tokenizer ---
        load_mixtral_model_and_tokenizer(args.model_name_or_path, args.num_gpus)
        if mixtral_model is None or mixtral_tokenizer is None:
            print("Fatal: Teacher model/tokenizer failed to load after attempt. Exiting.")
            return

        processed_task_ids = load_processed_task_ids(args.output_training_folder)
        
        print(f"Reading prompts from {args.input_prompts_file}...")
        step1_data_to_process = []
        # ... (Logic to read step1_data_to_process, skipping processed_task_ids, as before) ...
        try:
            with open(args.input_prompts_file, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    try:
                        data = json.loads(line)
                        if "task_id" in data and data["task_id"] not in processed_task_ids:
                            step1_data_to_process.append(data)
                    except json.JSONDecodeError: print(f"Warn: Skipping invalid JSON line: {line.strip()}")
        except Exception as e: print(f"Error reading input prompts file: {e}"); return

        if not step1_data_to_process: print("No new prompts to process."); return

        if args.num_samples is not None and args.num_samples > 0:
            if args.num_samples < len(step1_data_to_process):
                print(f"Processing sample of {args.num_samples} new prompts out of {len(step1_data_to_process)}.")
                step1_data_to_process = step1_data_to_process[:args.num_samples]
            else: print(f"Available new prompts ({len(step1_data_to_process)}) <= requested. Processing all.")
        else: print(f"Processing all {len(step1_data_to_process)} new prompts.")

        teacher_prompts_to_run = [entry["teacher_prompt_full"] for entry in step1_data_to_process]
        if not teacher_prompts_to_run: print("No prompts selected after filtering/sampling."); return

        generated_explanations_raw = get_mixtral_explanations_batched(
            teacher_prompts_to_run, args.batch_size, args.max_new_tokens
        )

        if len(generated_explanations_raw) != len(step1_data_to_process):
            print(f"Error: Mismatch explanations ({len(generated_explanations_raw)}) vs prompts ({len(step1_data_to_process)})."); return

        print(f"Post-processing and writing to folder: {args.output_training_folder} ...")
        current_output_file_handle: Optional[IO[str]] = None
        current_file_part_number = 0; samples_written_to_current_part = 0; final_pairs_written_this_run = 0
        existing_part_files = glob.glob(os.path.join(args.output_training_folder, "training_data_part_*.jsonl"))
        if existing_part_files:
            part_numbers = [int(re.search(r'part_(\d+)\.jsonl$', os.path.basename(f_path)).group(1)) for f_path in existing_part_files if re.search(r'part_(\d+)\.jsonl$', os.path.basename(f_path))]
            if part_numbers: current_file_part_number = max(part_numbers)
        
        # Inner try/finally for file handle
        try:
            for i, entry in enumerate(tqdm(step1_data_to_process, desc="Saving Final Data")):
                if args.checkpoint_every > 0 and (current_output_file_handle is None or samples_written_to_current_part >= args.checkpoint_every):
                    if current_output_file_handle: current_output_file_handle.close(); print(f"\nClosed: {current_output_filename}")
                    current_file_part_number += 1
                    current_output_filename = os.path.join(args.output_training_folder, f"training_data_part_{current_file_part_number:04d}.jsonl")
                    print(f"Opening new part file: {current_output_filename}")
                    current_output_file_handle = open(current_output_filename, "w", encoding="utf-8")
                    samples_written_to_current_part = 0
                elif current_output_file_handle is None and args.checkpoint_every == 0:
                    current_output_filename = os.path.join(args.output_training_folder, f"training_data_full.jsonl")
                    print(f"Opening output file: {current_output_filename}")
                    current_output_file_handle = open(current_output_filename, "w", encoding="utf-8") # Overwrite if exists, as resuming logic is based on task_ids

                student_prompt, task_id = entry["student_prompt_input"], entry["task_id"]
                raw_explanation = generated_explanations_raw[i]
                if raw_explanation == "GENERATION_ERROR": print(f"Warn: Skip task_id {task_id} due to gen error."); continue
                clean_explanation = post_process_explanation(raw_explanation, entry["teacher_prompt_full"])
                if not clean_explanation: print(f"Warn: Skip task_id {task_id} due to empty post-processed explanation."); continue
                
                final_training_pair = {"task_id": task_id, "input": student_prompt, "output": clean_explanation}
                current_output_file_handle.write(json.dumps(final_training_pair) + "\n")
                samples_written_to_current_part += 1
                final_pairs_written_this_run += 1
        finally:
             if current_output_file_handle and not current_output_file_handle.closed:
                 current_output_file_handle.close()
                 print(f"\nClosed final output file: {current_output_filename if 'current_output_filename' in locals() else 'output file'}")

        print(f"\nGenerated and saved {final_pairs_written_this_run} final training pairs in this run.")
        print(f"✅ Phase 2 training data saved in folder: {args.output_training_folder}")

    finally: # Top-level try...finally for model cleanup
        # --- Clean up model ---
        # No need for 'global' here if it's at the top of main
        if mixtral_model:
            del mixtral_model
            print("Mixtral model deleted from main.")
        if mixtral_tokenizer:
            del mixtral_tokenizer
            print("Mixtral tokenizer deleted from main.")
        
        mixtral_model = None # Ensure globals are reset
        mixtral_tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache emptied.")
        print("Cleanup attempt in main's finally block complete.")


if __name__ == "__main__":
    main()