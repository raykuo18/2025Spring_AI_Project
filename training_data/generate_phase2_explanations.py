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
import gc

# Common settings for generation
MAX_NEW_TOKENS_DEFAULT = 150

# Global model and tokenizer to load only once
mixtral_model = None
mixtral_tokenizer = None

def load_mixtral_model_and_tokenizer(model_name_or_path: str, num_gpus_expected: Optional[int] = None):
    global mixtral_model, mixtral_tokenizer
    if mixtral_model is not None and mixtral_tokenizer is not None:
        return
    print(f"Loading Mixtral model: {model_name_or_path}...")
    # ... (rest of model loading logic as in your previous full script) ...
    if num_gpus_expected is not None and (not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus_expected):
        print(f"Warning: CUDA not available or less than {num_gpus_expected} GPUs found. Attempting to load on available devices.")
    try:
        mixtral_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        if mixtral_tokenizer.pad_token is None:
            mixtral_tokenizer.pad_token = mixtral_tokenizer.eos_token
        mixtral_tokenizer.padding_side = "left"
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        mixtral_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=quantization_config, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, attn_implementation="sdpa")
        print(f"Mixtral 8x7B model successfully loaded. First param on: {next(mixtral_model.parameters()).device}")
    except Exception as e:
        print(f"Failed to load Mixtral model: {e}"); import traceback; traceback.print_exc(); mixtral_model = None; mixtral_tokenizer = None; raise


def get_mixtral_explanations_batched(teacher_prompts: List[str], batch_size: int, max_new_tokens: int, pbar_desc: str = "Mixtral Batch") -> List[str]:
    global mixtral_model, mixtral_tokenizer
    # ... (function content as in your previous full script) ...
    if mixtral_model is None or mixtral_tokenizer is None: raise RuntimeError("Mixtral model and tokenizer not loaded.")
    all_explanations = []
    num_prompts = len(teacher_prompts)
    for i in tqdm(range(0, num_prompts, batch_size), desc=pbar_desc, leave=False, ncols=100):
        batch_teacher_prompts = teacher_prompts[i:i + batch_size]
        practical_max_tokenization_length = getattr(mixtral_tokenizer, 'model_max_length', 4096)
        if practical_max_tokenization_length is None or practical_max_tokenization_length > 8192 : practical_max_tokenization_length = 8192
        inputs = mixtral_tokenizer(batch_teacher_prompts, return_tensors="pt", padding=True, truncation=True, max_length= practical_max_tokenization_length - max_new_tokens).to(mixtral_model.device)
        try:
            output_ids_batch = mixtral_model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=max_new_tokens, pad_token_id=mixtral_tokenizer.pad_token_id, eos_token_id=mixtral_tokenizer.eos_token_id, do_sample=True, temperature=0.6, top_p=0.9)
        except Exception as e_gen: print(f"Error during model.generate for batch {i}: {e_gen}"); all_explanations.extend(["GENERATION_ERROR"] * len(batch_teacher_prompts)); continue
        for j in range(output_ids_batch.shape[0]):
            prompt_len = inputs["attention_mask"][j].sum().item()
            generated_ids = output_ids_batch[j, prompt_len:]; explanation = mixtral_tokenizer.decode(generated_ids, skip_special_tokens=True); all_explanations.append(explanation)
    return all_explanations


def post_process_explanation(raw_text: str, teacher_task_instruction: Optional[str]=None) -> str:
    # ... (function content as in your previous full script) ...
    processed_text = raw_text.strip()
    common_boilerplate = ["explanation: \[assistant\]", "explanation:", "\[assistant\]", "okay, here's an explanation:", "sure, i can explain that.", "here is the explanation:", "here's a concise explanation:", "the explanation is as follows:"]
    for bp in common_boilerplate:
        escaped_bp = re.escape(bp).replace(r'\\\[', r'\[').replace(r'\\\]', r'\]')
        match = re.match(rf"^\s*{escaped_bp}\s*", processed_text, re.IGNORECASE)
        if match: processed_text = processed_text[match.end():].strip(); break
    prompt_echo_phrases = ["e.g., central control, development, king safety, pawn structure) OR immediate tactical ideas (e.g., threats, defenses, piece activation, opening ideas).", "opening ideas).", "piece activation, opening ideas).", "resulted in 'Capture'. Explain concisely (1-2 sentences) what this event means or achieves in this specific position.", "resulted in 'Check'. Explain concisely (1-2 sentences) what this event means or achieves in this specific position."]
    if teacher_task_instruction:
        first_line_of_task = teacher_task_instruction.split('\n')[0].strip()
        if first_line_of_task and processed_text.lower().startswith(first_line_of_task.lower()): processed_text = processed_text[len(first_line_of_task):].strip()
        else:
            for phrase in prompt_echo_phrases:
                escaped_phrase = re.escape(phrase)
                match = re.match(rf"^\s*{escaped_phrase}\s*", processed_text, re.IGNORECASE)
                if match: processed_text = processed_text[match.end():].strip(); break
    sentences = re.split(r'(?<=[.!?])\s+', processed_text); max_sentences = 4 
    if sentences and sentences[0]: processed_text = " ".join(sentences[:max_sentences])
    else: processed_text = "" if not raw_text.strip() else raw_text.strip()
    if processed_text and processed_text[-1] not in ".!?": processed_text += "."
    processed_text = re.sub(r'\s\s+', ' ', processed_text).strip(); processed_text = processed_text.replace("\n", " ").strip()
    return processed_text

def load_processed_task_ids(output_folder_path: str) -> Set[str]:
    processed_ids = set()
    if not os.path.isdir(output_folder_path): return processed_ids
    
    # Scan all potential output file patterns
    file_patterns = [
        os.path.join(output_folder_path, "training_data_slice_*_part_*.jsonl"),
        os.path.join(output_folder_path, "training_data_slice_*_all_current_run.jsonl"),
        os.path.join(output_folder_path, "training_data_part_*.jsonl"), # Legacy from previous version
        os.path.join(output_folder_path, "training_data_full_current_run.jsonl") # Legacy
    ]
    files_to_check = []
    for pattern in file_patterns:
        files_to_check.extend(glob.glob(pattern))
    
    if not files_to_check: print(f"No existing output files found in {output_folder_path} matching patterns."); return processed_ids
    
    unique_files = sorted(list(set(files_to_check))) # Avoid processing same file twice if multiple patterns match
    print(f"Checking {len(unique_files)} existing output file(s) for processed tasks...")
    for file_path in unique_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try: data = json.loads(line); processed_ids.add(data["task_id"])
                    except (json.JSONDecodeError, KeyError): pass 
        except Exception as e: print(f"Error reading {file_path}: {e}. Continuing...")
            
    print(f"Found {len(processed_ids)} already processed task_ids from existing output files."); return processed_ids

def main():
    global mixtral_model, mixtral_tokenizer
    parser = argparse.ArgumentParser(description="Step 2: Run Mixtral, save to checkpointed files with slicing.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input-prompts-file", required=True, help="JSONL file from Step 1.")
    parser.add_argument("-o", "--output-training-folder", required=True, help="FOLDER for final training data parts.")
    parser.add_argument("--model-name-or-path", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1", help="Mixtral model.")
    parser.add_argument("--batch-size", type=int, default=4, help="Mixtral inference batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS_DEFAULT, help="Max new tokens for Mixtral.")
    parser.add_argument("--num-gpus", type=int, default=None, help="Num GPUs. Default: auto.")
    # <<< NEW/MODIFIED SLICING ARGUMENTS >>>
    parser.add_argument("--slice-start-index", type=int, default=0,
                        help="0-based global start index in the input prompts file for this job.")
    parser.add_argument("--slice-num-samples", type=int, default=None,
                        help="Maximum number of samples for this job to process from its slice_start_index. Default: all from start index.")
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Save a new part file every N processed samples by this job. 0 for single output file for this job's slice per run.")
    args = parser.parse_args()

    if not os.path.exists(args.input_prompts_file): parser.error(f"Input file not found: {args.input_prompts_file}")
    if not os.path.exists(args.output_training_folder): os.makedirs(args.output_training_folder, exist_ok=True)
    elif not os.path.isdir(args.output_training_folder): parser.error(f"Output path not a directory: {args.output_training_folder}")

    try:
        load_mixtral_model_and_tokenizer(args.model_name_or_path, args.num_gpus)
        if mixtral_model is None or mixtral_tokenizer is None: return

        # --- Read ALL prompts from Step 1 output ---
        print(f"Reading all available prompts from {args.input_prompts_file}...")
        all_available_prompts = []
        try:
            with open(args.input_prompts_file, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    try: all_available_prompts.append(json.loads(line))
                    except json.JSONDecodeError: print(f"Warn: Skipping invalid JSON line in prompts file: {line.strip()}")
        except Exception as e: print(f"Error reading input prompts file: {e}"); return
        
        if not all_available_prompts: print("No prompts found in input file."); return
        print(f"Total prompts available in input file: {len(all_available_prompts)}")

        # --- Determine this job's specific slice from all_available_prompts ---
        job_slice_start = args.slice_start_index
        if job_slice_start < 0: job_slice_start = 0
        
        if args.slice_num_samples is not None and args.slice_num_samples > 0:
            job_slice_end_exclusive = min(job_slice_start + args.slice_num_samples, len(all_available_prompts))
        else: # Process till end of file from start_index
            job_slice_end_exclusive = len(all_available_prompts)

        if job_slice_start >= len(all_available_prompts):
            print(f"Slice start index ({job_slice_start}) is beyond the number of available prompts ({len(all_available_prompts)}). No work to do for this job."); return
            
        prompts_assigned_to_this_job = all_available_prompts[job_slice_start:job_slice_end_exclusive]
        
        if not prompts_assigned_to_this_job: print(f"No prompts assigned to this job's slice ({job_slice_start}-{job_slice_end_exclusive-1})."); return
        print(f"This job is assigned {len(prompts_assigned_to_this_job)} prompts (global indices {job_slice_start} to {job_slice_end_exclusive-1}).")

        # --- Load globally processed task IDs to filter this job's assigned slice ---
        processed_task_ids = load_processed_task_ids(args.output_training_folder)
        
        step1_data_to_process = [
            entry for entry in prompts_assigned_to_this_job 
            if "task_id" in entry and entry["task_id"] not in processed_task_ids
        ]

        if not step1_data_to_process: print("No new prompts to process for this job's slice (all might be processed already)."); return
        print(f"This job will process {len(step1_data_to_process)} new prompts from its assigned slice.")

        # --- Main Processing Loop - Chunked for Checkpointing ---
        final_pairs_written_this_run = 0
        current_output_file_handle: Optional[IO[str]] = None
        current_output_filename = ""
        
        # Filename prefix for this job's output, reflecting the global slice it's responsible for
        job_slice_name_prefix = f"data_slice_{job_slice_start}_to_{job_slice_end_exclusive-1}"

        process_all_in_one_chunk_for_job = (args.checkpoint_every <= 0)
        chunk_size = len(step1_data_to_process) if process_all_in_one_chunk_for_job else args.checkpoint_every
        if chunk_size == 0 and len(step1_data_to_process) > 0 : chunk_size = len(step1_data_to_process)
        if chunk_size == 0: print("No data to process after chunk calculation."); return

        outer_pbar = tqdm(total=len(step1_data_to_process), unit="prompt", desc=f"Job Slice {job_slice_start}-{job_slice_end_exclusive-1}", ncols=100)
        
        # Inner try/finally for file handle
        try:
            for i in range(0, len(step1_data_to_process), chunk_size):
                current_chunk_data_entries = step1_data_to_process[i:i + chunk_size]
                if not current_chunk_data_entries: continue

                teacher_prompts_for_chunk = [entry["teacher_prompt_full"] for entry in current_chunk_data_entries]
                
                # Indices for this part file are relative to the current job's new items
                local_part_start_index = i
                local_part_end_index = min(i + chunk_size - 1, len(step1_data_to_process) - 1)

                pbar_desc_batch = f"Mixtral (Job Slice {job_slice_start}-{job_slice_end_exclusive-1}, Prompts {local_part_start_index}-{local_part_end_index} of this job's new items)"
                generated_explanations_for_chunk = get_mixtral_explanations_batched(
                    teacher_prompts_for_chunk, args.batch_size, args.max_new_tokens, pbar_desc=pbar_desc_batch
                )

                if len(generated_explanations_for_chunk) != len(current_chunk_data_entries):
                    print(f"Error: Mismatch explanations for chunk {pbar_desc_batch}. Skipping."); outer_pbar.update(len(current_chunk_data_entries)); continue

                if current_output_file_handle and not process_all_in_one_chunk_for_job : current_output_file_handle.close(); print(f"\nClosed previous part: {current_output_filename}")
                
                if process_all_in_one_chunk_for_job:
                    current_output_filename = os.path.join(args.output_training_folder, f"{job_slice_name_prefix}_all_current_run.jsonl")
                    open_mode = "w" if i == 0 else "a" # Write for first (and only) chunk, should be 'w'
                    if i == 0: print(f"\nOpening single output file for this job's slice: {current_output_filename}")
                else: # Part files
                    current_output_filename = os.path.join(args.output_training_folder, f"{job_slice_name_prefix}_part_{local_part_start_index}_to_{local_part_end_index}.jsonl")
                    print(f"\nOpening new part file: {current_output_filename}")
                    open_mode = "w" # Each part file is new

                if process_all_in_one_chunk_for_job and current_output_file_handle and not current_output_file_handle.closed and open_mode == "a":
                    pass # Keep using same handle
                else:
                    current_output_file_handle = open(current_output_filename, open_mode, encoding="utf-8")

                items_written_in_this_chunk = 0
                for j, entry in enumerate(current_chunk_data_entries):
                    student_prompt, task_id = entry["student_prompt_input"], entry["task_id"]
                    raw_explanation = generated_explanations_for_chunk[j]
                    if raw_explanation == "GENERATION_ERROR": print(f"Warn: Skip {task_id} (gen error)."); outer_pbar.update(1); continue
                    teacher_task_instr = entry.get("teacher_task_instruction") 
                    clean_explanation = post_process_explanation(raw_explanation, teacher_task_instr)
                    if not clean_explanation: print(f"Warn: Skip {task_id} (empty post-proc)."); outer_pbar.update(1); continue
                    final_training_pair = {"task_id": task_id, "input": student_prompt, "teacher_raw_output": raw_explanation, "output": clean_explanation}
                    if current_output_file_handle:
                        current_output_file_handle.write(json.dumps(final_training_pair) + "\n")
                        final_pairs_written_this_run += 1; items_written_in_this_chunk +=1
                    outer_pbar.update(1)

                if current_output_file_handle and not process_all_in_one_chunk_for_job:
                    current_output_file_handle.flush(); os.fsync(current_output_file_handle.fileno())
                    print(f"Checkpoint: Flushed {items_written_in_this_chunk} items to {current_output_filename}. Total this run: {final_pairs_written_this_run}")
        finally:
             if current_output_file_handle and not current_output_file_handle.closed:
                 current_output_file_handle.close()
                 print(f"\nClosed final output file: {current_output_filename if 'current_output_filename' in locals() else 'output file'}")
        if 'outer_pbar' in locals() and outer_pbar: outer_pbar.close()
        
    finally: # Top-level try...finally for model cleanup
        if mixtral_model: del mixtral_model; mixtral_model = None; gc.collect()
        if mixtral_tokenizer: del mixtral_tokenizer; mixtral_tokenizer = None; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print("Mixtral model, tokenizer cleared, and CUDA cache (if used) emptied.")

    print(f"\nGenerated and saved {final_pairs_written_this_run} final training pairs in this run for slice {job_slice_start}-{job_slice_end_exclusive-1}.")
    print(f"✅ Phase 2 training data parts saved in folder: {args.output_training_folder}")

if __name__ == "__main__":
    main()