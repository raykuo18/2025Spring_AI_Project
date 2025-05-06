#!/usr/bin/env python3

import json
import re
import argparse
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from typing import List, Dict, Optional, Set, IO # Added IO for file handle type hint
import glob # For finding part files

# Common settings for generation
MAX_NEW_TOKENS_DEFAULT = 150

# Global model and tokenizer to load only once
mixtral_model = None
mixtral_tokenizer = None

def load_mixtral_model_and_tokenizer(model_name_or_path: str, num_gpus_expected: Optional[int] = None):
    global mixtral_model, mixtral_tokenizer
    # ... (function content as in the previous version) ...
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
    # ... (function content as in the previous version) ...
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
            # Calculate prompt length more robustly using attention mask
            prompt_len = inputs["attention_mask"][j].sum().item()
            generated_ids = output_ids_batch[j, prompt_len:]
            explanation = mixtral_tokenizer.decode(generated_ids, skip_special_tokens=True)
            all_explanations.append(explanation)
    return all_explanations


def post_process_explanation(text: str, original_prompt: Optional[str]=None) -> str:
    # ... (function content as in the previous version, or expanded based on Mixtral outputs) ...
    processed_text = text.strip()
    boilerplate_starts = [
        "Explanation:", "Okay, here's an explanation:", "Sure, I can explain that.", "Here is the explanation:",
        "Here's a concise explanation:", "The explanation is as follows:"
    ]
    for bp_start in boilerplate_starts:
        if processed_text.lower().startswith(bp_start.lower()):
            processed_text = processed_text[len(bp_start):].strip()
    if processed_text.startswith("[ASSISTANT]"): # If the model included the turn token
        processed_text = processed_text[len("[ASSISTANT]"):].strip()
    
    # Attempt to remove prompt repetition if the model echoes input
    # This is a heuristic and might need careful adjustment
    if original_prompt:
        # Check if a significant portion of the start of processed_text matches end of original_prompt
        # (especially if original_prompt ends with a clear marker like "Explanation:\n[ASSISTANT]")
        marker = "Explanation:\n[ASSISTANT]" # Common marker in our teacher prompts
        idx = original_prompt.rfind(marker)
        if idx != -1:
            prompt_tail = original_prompt[idx:]
            if processed_text.startswith(prompt_tail):
                 processed_text = processed_text[len(prompt_tail):].strip()
            elif processed_text.startswith(marker.split("\n")[-1].strip()): # Just "[ASSISTANT]"
                 processed_text = processed_text[len(marker.split("\n")[-1].strip()):].strip()


    sentences = re.split(r'(?<=[.!?])\s+', processed_text)
    max_sentences = 4
    if len(sentences) > max_sentences:
        processed_text = " ".join(sentences[:max_sentences])
    if processed_text and processed_text[-1] not in ".!?":
        processed_text += "."
    processed_text = re.sub(r'\s\s+', ' ', processed_text).strip()
    processed_text = processed_text.replace("\n", " ").strip()
    return processed_text

# <<< MODIFIED FUNCTION to scan folder >>>
def load_processed_task_ids(output_folder_path: str) -> Set[str]:
    """Loads task_ids from all existing part files in the output folder."""
    processed_ids = set()
    if not os.path.isdir(output_folder_path):
        return processed_ids # Folder doesn't exist yet

    # Find all part files, e.g., training_data_part_*.jsonl
    # Sort them to process in order, though set order doesn't matter
    part_files = sorted(glob.glob(os.path.join(output_folder_path, "training_data_part_*.jsonl")))
    
    if not part_files:
        print(f"No existing part files found in {output_folder_path}.")
        return processed_ids

    print(f"Found existing output part files: {part_files}. Checking for processed tasks...")
    for part_file in part_files:
        try:
            with open(part_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "task_id" in data:
                            processed_ids.add(data["task_id"])
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line in existing output file {part_file}: {line.strip()}")
        except Exception as e:
            print(f"Error reading existing output part file {part_file}: {e}. Continuing...")
            
    print(f"Found {len(processed_ids)} already processed task_ids from existing part files.")
    return processed_ids

def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Run Mixtral inference, save to checkpointed files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input-prompts-file", required=True,
                        help="Path to the JSONL file from Step 1 (containing teacher_prompt_full).")
    # <<< MODIFIED ARGUMENT >>>
    parser.add_argument("-o", "--output-training-folder", required=True,
                        help="Path to the FOLDER to save final Phase 2 training data part files.")
    parser.add_argument("--model-name-or-path", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        help="Path or Hugging Face name of the Mixtral model.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for Mixtral inference.")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS_DEFAULT,
                        help="Max new tokens for Mixtral to generate.")
    parser.add_argument("--num-gpus", type=int, default=None, help="Specify number of GPUs. Default: auto.")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Max number of NEW prompts to process. If None, process all new prompts.")
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Save a new part file every N processed samples. 0 to disable intermediate parts (only one final file).")

    args = parser.parse_args()

    if not os.path.exists(args.input_prompts_file):
        parser.error(f"Input prompts file not found: {args.input_prompts_file}")

    # Ensure output folder exists (it will be created if it doesn't)
    if not os.path.exists(args.output_training_folder):
        print(f"Output folder {args.output_training_folder} does not exist. It will be created.")
        os.makedirs(args.output_training_folder, exist_ok=True)
    elif not os.path.isdir(args.output_training_folder):
        parser.error(f"Output path {args.output_training_folder} exists but is not a directory.")


    # --- Load Mixtral Model and Tokenizer ---
    try:
        load_mixtral_model_and_tokenizer(args.model_name_or_path, args.num_gpus)
    except Exception as e: print(f"Fatal: Could not load teacher model. Exiting. Error: {e}"); return
    if mixtral_model is None or mixtral_tokenizer is None: print("Fatal: Teacher model/tokenizer failed. Exiting."); return

    # --- Load already processed task IDs for resuming ---
    processed_task_ids = load_processed_task_ids(args.output_training_folder)
    
    # --- Read Prompts from Step 1 Output, skipping processed ones ---
    print(f"Reading prompts from {args.input_prompts_file}...")
    step1_data_to_process = []
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

    # --- Apply --num-samples limit to NEW prompts ---
    if args.num_samples is not None and args.num_samples > 0:
        if args.num_samples < len(step1_data_to_process):
            print(f"Processing a sample of {args.num_samples} new prompts out of {len(step1_data_to_process)} available.")
            step1_data_to_process = step1_data_to_process[:args.num_samples]
        else:
            print(f"Available new prompts ({len(step1_data_to_process)}) <= requested ({args.num_samples}). Processing all available.")
    else:
        print(f"Processing all {len(step1_data_to_process)} available new prompts.")

    teacher_prompts_to_run = [entry["teacher_prompt_full"] for entry in step1_data_to_process]
    if not teacher_prompts_to_run: print("No prompts selected after filtering/sampling."); return

    # --- Get Explanations from Mixtral ---
    generated_explanations_raw = get_mixtral_explanations_batched(
        teacher_prompts_to_run, args.batch_size, args.max_new_tokens
    )

    if len(generated_explanations_raw) != len(step1_data_to_process):
        print(f"Error: Mismatch in explanations ({len(generated_explanations_raw)}) vs prompts ({len(step1_data_to_process)})."); return

    # --- Post-Process and Write Final Training Data with Checkpointing ---
    print(f"Post-processing and writing to folder: {args.output_training_folder} ...")
    
    current_output_file_handle: Optional[IO[str]] = None
    current_file_part_number = 0
    samples_written_to_current_part = 0
    final_pairs_written_this_run = 0

    # Determine starting part number by checking existing files
    existing_part_files = glob.glob(os.path.join(args.output_training_folder, "training_data_part_*.jsonl"))
    if existing_part_files:
        part_numbers = []
        for f_path in existing_part_files:
            match = re.search(r'part_(\d+)\.jsonl$', os.path.basename(f_path))
            if match:
                part_numbers.append(int(match.group(1)))
        if part_numbers:
            current_file_part_number = max(part_numbers) # Start from next part number if continuing

    try:
        for i, entry in enumerate(tqdm(step1_data_to_process, desc="Saving Final Data")):
            # --- Open new part file if needed ---
            if args.checkpoint_every > 0 and (current_output_file_handle is None or samples_written_to_current_part >= args.checkpoint_every):
                if current_output_file_handle:
                    current_output_file_handle.close()
                    print(f"\nClosed part file: {current_output_filename}")
                
                current_file_part_number += 1
                current_output_filename = os.path.join(args.output_training_folder, f"training_data_part_{current_file_part_number:04d}.jsonl")
                print(f"Opening new part file: {current_output_filename}")
                # Important: Open in "w" mode for new part files as resuming is handled by skipping prompts.
                # If script is stopped and resumed, it will create a NEW part file for the remaining prompts.
                current_output_file_handle = open(current_output_filename, "w", encoding="utf-8")
                samples_written_to_current_part = 0
            elif current_output_file_handle is None and args.checkpoint_every == 0: # No checkpointing, single file logic
                current_output_filename = os.path.join(args.output_training_folder, f"training_data_full.jsonl")
                print(f"Opening output file: {current_output_filename}")
                current_output_file_handle = open(current_output_filename, "w", encoding="utf-8")


            student_prompt = entry["student_prompt_input"]
            task_id = entry["task_id"]
            raw_explanation = generated_explanations_raw[i]

            if raw_explanation == "GENERATION_ERROR": print(f"Warn: Skip task_id {task_id} due to gen error."); continue
            clean_explanation = post_process_explanation(raw_explanation, original_prompt=entry["teacher_prompt_full"])
            if not clean_explanation: print(f"Warn: Skip task_id {task_id} due to empty post-processed explanation."); continue
                
            final_training_pair = {"task_id": task_id, "input": student_prompt, "output": clean_explanation}
            current_output_file_handle.write(json.dumps(final_training_pair) + "\n")
            samples_written_to_current_part += 1
            final_pairs_written_this_run += 1
        
    finally: # Ensure the last opened file is closed
        if current_output_file_handle and not current_output_file_handle.closed:
            current_output_file_handle.close()
            print(f"\nClosed final part file: {current_output_filename if 'current_output_filename' in locals() else 'output file'}")

    print(f"\nGenerated and saved {final_pairs_written_this_run} final training pairs in this run.")
    print(f"✅ Phase 2 training data (with Mixtral labels) saved in folder: {args.output_training_folder}")

    # --- Clean up model ---
    global mixtral_model, mixtral_tokenizer
    if mixtral_model: del mixtral_model
    if mixtral_tokenizer: del mixtral_tokenizer
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("Mixtral model and tokenizer cleared from memory.")

if __name__ == "__main__":
    main()
