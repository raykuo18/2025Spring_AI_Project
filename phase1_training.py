#!/usr/bin/env python3

import argparse
import os
import json
import random
import numpy as np
import torch
import sys # For sys.exit()
from typing import Optional
# Hugging Face Libraries
from datasets import load_dataset, DatasetDict, Dataset 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig # For QLoRA
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training, # For QLoRA
    TaskType
)
# tqdm is implicitly used by datasets.map and Trainer, but not directly imported here.

# --- Model Configuration Mapping ---
MODEL_CONFIGS = {
    "TinyLLaMA": {
        "hf_model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"], # Common for Llama
        "add_eos_token": True, # Typically for chat models
    },
    "Gemma-2B": {
        "hf_model_name": "google/gemma-2b", # Use "google/gemma-2b-it" for instruct version
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"], # Common for Gemma
        "add_eos_token": True,
    },
    "Phi-2": {
        "hf_model_name": "microsoft/phi-2",
        "target_modules": ["q_proj", "v_proj", "k_proj", "dense"], # More common for Phi-2 architecture
        "add_eos_token": True,
    }
}

def set_seed(seed: int):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune or only tokenize for Causal LM with LoRA.")
    # Model and Data Arguments
    parser.add_argument("--model_name", type=str, required=True, choices=MODEL_CONFIGS.keys(), help="Name of the base model.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training JSONL.")
    parser.add_argument("--val_file", type=str, required=True, help="Path to validation JSONL.")
    parser.add_argument("--test_file", type=str, default=None, help="Optional: Path to test JSONL for final evaluation.")
    parser.add_argument("--base_model_cache_dir", type=str, default="./pretrained_cache", help="Cache dir for Hugging Face models.")
    parser.add_argument("--output_dir", type=str, required=True, help="Dir for LoRA adapter & training artifacts (if training).")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length.")

    # Tokenized Data Caching & Sub-sampling Arguments
    parser.add_argument("--tokenized_data_path", type=str, default=None,
                        help="Base path to save/load tokenized datasets. Each split (train/val/test) will be a subdirectory.")
    parser.add_argument("--overwrite_tokenized_cache", action="store_true",
                        help="Force re-tokenization even if cached tokenized data exists.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max number of training samples to load and tokenize.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Max number of validation samples to load and tokenize.")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Max number of test samples to load and tokenize.")
    
    parser.add_argument("--tokenize_only", action="store_true",
                        help="If set, load data, tokenize, save tokenized data (if path provided), and exit.")
    
    # LoRA Arguments
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout.")

    # Training Hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Train batch size per device.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Eval batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type.")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit", help="Optimizer.")

    # Logging and Saving
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X steps.")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every X steps.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit total checkpoints.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Report metrics to (e.g. tensorboard, wandb, none).")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load base model in 4-bit (QLoRA).")
    parser.add_argument("--use_flash_attention_2", action="store_true", help="Use Flash Attention 2 (if available and not using 4-bit).")

    args = parser.parse_args()
    set_seed(args.seed)

    # --- 1. Load Tokenizer ---
    model_config_details = MODEL_CONFIGS[args.model_name]
    hf_model_name = model_config_details["hf_model_name"]
    add_eos_token_to_output = model_config_details.get("add_eos_token", True)

    print(f"Loading tokenizer for {hf_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name,
        cache_dir=args.base_model_cache_dir,
        use_fast=True,
        trust_remote_code=True # Some models like Phi-2 might require this
    )
    if tokenizer.pad_token is None:
        print("Tokenizer missing pad_token, setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Standard for Causal LM with Trainer

    # --- 2. Dataset Preparation ---
    tokenized_train_dataset: Optional[Dataset] = None
    tokenized_eval_dataset: Optional[Dataset] = None
    tokenized_test_dataset: Optional[Dataset] = None
    
    sanitized_model_name = hf_model_name.split('/')[-1].replace('-', '_')
    
    datasets_to_process_config = {
        "train": {"file": args.train_file, "max_samples": args.max_train_samples, "cache_suffix": "train"},
        "validation": {"file": args.val_file, "max_samples": args.max_eval_samples, "cache_suffix": "eval"},
    }
    if args.test_file:
        datasets_to_process_config["test"] = {"file": args.test_file, "max_samples": args.max_test_samples, "cache_suffix": "test"}

    tokenized_datasets_loaded = {}

    for split_name, config in datasets_to_process_config.items():
        raw_file_path = config["file"]
        max_samples_for_split = config["max_samples"]
        cache_suffix_for_split = config["cache_suffix"] # e.g., "train", "eval", "test"
        cache_path_for_split = None

        if args.tokenized_data_path:
            # Incorporate max_samples into the cache path name if it's applied at load time
            subset_info_suffix = f"_top{max_samples_for_split}" if max_samples_for_split is not None else ""
            cache_dir_for_split = os.path.join(args.tokenized_data_path, f"{sanitized_model_name}_seq{args.max_seq_length}_{cache_suffix_for_split}{subset_info_suffix}")
            cache_path_for_split = cache_dir_for_split
            
            if not args.overwrite_tokenized_cache and os.path.exists(os.path.join(cache_path_for_split, "dataset_info.json")):
                print(f"Loading tokenized {split_name} data from cache: {cache_path_for_split}")
                try:
                    tokenized_datasets_loaded[split_name] = Dataset.load_from_disk(cache_path_for_split)
                    print(f"Loaded {len(tokenized_datasets_loaded[split_name])} samples for {split_name} from cache.")
                    continue # Skip to next split if loaded from cache
                except Exception as e:
                    print(f"Failed to load {split_name} cache: {e}. Will re-tokenize.")
        
        # If not loaded from cache, tokenize
        print(f"Preparing to tokenize {split_name} data from {raw_file_path}...")
        
        current_data_file_dict_for_load = {split_name: raw_file_path} # For load_dataset data_files arg
        slicing_instruction = split_name # Default to loading the whole named split from the file
        
        if max_samples_for_split is not None and max_samples_for_split > 0:
            slicing_instruction = f"{split_name}[:{max_samples_for_split}]"
            print(f"Will load a maximum of {max_samples_for_split} raw samples for {split_name} set using split='{slicing_instruction}'.")
        
        # Load raw data (potentially sliced)
        # The key in data_files (split_name) must match the split name used for slicing
        try:
            raw_dataset_this_split = load_dataset("json", data_files=current_data_file_dict_for_load, split=slicing_instruction)
        except Exception as e:
            print(f"Error loading raw {split_name} data: {e}. Skipping this split.")
            continue


        def preprocess_function(examples):
            inputs_text = [] 
            prompts_text = examples["input"]
            outputs_text = examples["output"]

            for i in range(len(prompts_text)):
                # Ensure output is a string, handle None or other types gracefully
                output_str = str(outputs_text[i]) if outputs_text[i] is not None else ""
                text = prompts_text[i] + output_str 
                if add_eos_token_to_output: text += tokenizer.eos_token
                inputs_text.append(text)

            model_inputs = tokenizer(inputs_text, max_length=args.max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
            labels = model_inputs["input_ids"].clone()

            for i in range(len(prompts_text)):
                prompt_only_text = prompts_text[i]
                # Tokenize prompt to get its length including any special tokens tokenizer might add at start of sequence
                # Pass add_special_tokens=True if your main tokenization of combined text would include them for the prompt part
                tokenized_prompt = tokenizer(prompt_only_text, max_length=args.max_seq_length, truncation=True, add_special_tokens=True) 
                prompt_tokens_length = torch.sum(torch.tensor(tokenized_prompt['attention_mask'])).item()
                
                mask_len = min(prompt_tokens_length, labels.shape[1])
                labels[i, :mask_len] = -100
            
            model_inputs["labels"] = labels
            return model_inputs

        num_proc = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
        print(f"Tokenizing {split_name} set ({len(raw_dataset_this_split)} raw samples) with {num_proc} processes...")
        tokenized_split = raw_dataset_this_split.map(
            preprocess_function, batched=True, remove_columns=raw_dataset_this_split.column_names,
            num_proc=num_proc, desc=f"Tokenizing {split_name} set"
        )
        tokenized_datasets_loaded[split_name] = tokenized_split

        if cache_path_for_split:
            print(f"Saving tokenized {split_name} data to cache: {cache_path_for_split}")
            os.makedirs(cache_path_for_split, exist_ok=True)
            tokenized_split.save_to_disk(cache_path_for_split)

    tokenized_train_dataset = tokenized_datasets_loaded.get("train")
    tokenized_eval_dataset = tokenized_datasets_loaded.get("validation")
    tokenized_test_dataset = tokenized_datasets_loaded.get("test")

    # Final checks after all attempts to load or tokenize
    if not tokenized_train_dataset:
        print("Error: Training dataset is missing after tokenization/loading. Exiting.")
        sys.exit(1)
    if not tokenized_eval_dataset:
        print("Warning: Validation dataset is missing after tokenization/loading.")
        # Allow training without eval, but Trainer might complain or need eval_dataset=None
    if args.test_file and not tokenized_test_dataset:
        print("Warning: Test file specified but test dataset could not be prepared.")


    print(f"Final train dataset size: {len(tokenized_train_dataset) if tokenized_train_dataset else 0}")
    print(f"Final validation dataset size: {len(tokenized_eval_dataset) if tokenized_eval_dataset else 0}")
    if tokenized_test_dataset: print(f"Final test dataset size: {len(tokenized_test_dataset)}")


    if args.tokenize_only:
        if args.tokenized_data_path: print(f"Tokenization complete. Datasets saved in subdirectories of: {args.tokenized_data_path}")
        else: print("Tokenization complete. No --tokenized_data_path provided, so data is not persistently saved by this script (datasets library might still use its own default cache).")
        sys.exit(0)

    # --- If not tokenize_only, proceed to load full model and train ---
    if not args.output_dir: parser.error("--output_dir is required if not in --tokenize_only mode.")
    if not tokenized_train_dataset: # Should have exited earlier, but defensive check
        print("Error: Training data not available for training. Exiting.")
        sys.exit(1)


    target_modules = model_config_details["target_modules"]
    quantization_config = None
    if args.load_in_4bit:
        print("Loading model in 4-bit (QLoRA style) for training...")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                             bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                                             bnb_4bit_use_double_quant=True)
    print(f"Loading base model for training: {hf_model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name, cache_dir=args.base_model_cache_dir, quantization_config=quantization_config,
        torch_dtype="auto", device_map="auto", trust_remote_code=True,
        attn_implementation="flash_attention_2" if args.use_flash_attention_2 and quantization_config is None else "sdpa",
    )
    if model.config.pad_token_id is None: # Ensure model config also has pad token id
        model.config.pad_token_id = tokenizer.eos_token_id # tokenizer.pad_token was set to eos_token
    if args.load_in_4bit: 
        model = prepare_model_for_kbit_training(model)

    # --- LoRA Configuration ---
    print("Setting up LoRA configuration...")
    lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=target_modules, lora_dropout=args.lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Training ---
    print("Setting up Training Arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size, 
        per_device_eval_batch_size=args.per_device_eval_batch_size if tokenized_eval_dataset else args.per_device_train_batch_size, # Fallback if no eval_dataset
        gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio, lr_scheduler_type=args.lr_scheduler_type, optim=args.optim,
        logging_dir=os.path.join(args.output_dir, "logs"), logging_steps=args.logging_steps,
        evaluation_strategy="steps" if tokenized_eval_dataset and args.eval_steps > 0 else "no", 
        eval_steps=args.eval_steps if tokenized_eval_dataset and args.eval_steps > 0 else None,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=args.save_total_limit,
        load_best_model_at_end=True if tokenized_eval_dataset and args.eval_steps > 0 else False,
        metric_for_best_model="eval_loss" if tokenized_eval_dataset and args.eval_steps > 0 else None,
        greater_is_better=False if tokenized_eval_dataset and args.eval_steps > 0 else None,
        report_to=args.report_to.split(',') if args.report_to and args.report_to != "none" else None,
        fp16=torch.cuda.is_available() and not args.load_in_4bit and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported() and not args.load_in_4bit, seed=args.seed,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=tokenized_train_dataset, 
        eval_dataset=tokenized_eval_dataset, # Will be None if no val_file or val_dataset failed
        tokenizer=tokenizer, 
        data_collator=data_collator
    )
    print("Starting training...")
    trainer.train()
    print("Training complete.")
    
    # --- Evaluate on Test Set (if provided) ---
    if tokenized_test_dataset:
        print("\nEvaluating on the test set...")
        test_results = trainer.evaluate(eval_dataset=tokenized_test_dataset, metric_key_prefix="test")
        print(f"Test Set Metrics: {test_results}")
        test_results_path = os.path.join(args.output_dir, "test_results.json")
        with open(test_results_path, "w") as f: json.dump(test_results, f, indent=4)
        print(f"Test results saved to {test_results_path}")

    # --- Save Final LoRA Adapter ---
    final_adapter_path = os.path.join(args.output_dir, "final_lora_adapter")
    print(f"Saving final LoRA adapter to {final_adapter_path}...")
    model.save_pretrained(final_adapter_path)
    # Optionally save tokenizer if any modifications were made, though usually not needed for LoRA
    # tokenizer.save_pretrained(final_adapter_path) 
    print("LoRA adapter saved successfully.")
    print(f"Base model was: {hf_model_name}")


if __name__ == "__main__":
    main()