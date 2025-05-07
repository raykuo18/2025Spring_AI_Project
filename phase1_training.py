#!/usr/bin/env python3

import argparse
import os
import json
import random
import numpy as np
import torch
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets # Keep concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import sys
import glob # Keep glob for robust cache loading

# --- Model Configuration Mapping ---
MODEL_CONFIGS = {
    "TinyLLaMA": {
        "hf_model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "add_eos_token": True,
    },
    "Gemma-2B": {
        "hf_model_name": "google/gemma-2b",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "add_eos_token": True,
    },
    "Phi-2": {
        "hf_model_name": "microsoft/phi-2",
        "target_modules": ["q_proj", "v_proj", "k_proj", "dense"],
        "add_eos_token": True,
    }
}

def set_seed(seed: int):
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
    parser.add_argument("--test_file", type=str, default=None, help="Optional: Path to test JSONL for final evaluation.") # ADDED
    parser.add_argument("--base_model_cache_dir", type=str, default="./pretrained_cache", help="Cache dir for Hugging Face models.")
    parser.add_argument("--output_dir", type=str, required=True, help="Dir for LoRA adapter & training artifacts (if training).")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length.")

    # Tokenized Data Caching & Sub-sampling Arguments
    parser.add_argument("--tokenized_data_path", type=str, default=None,
                        help="Base path to save/load tokenized datasets. Each split (train/val/test) will be a subdirectory.")
    parser.add_argument("--overwrite_tokenized_cache", action="store_true",
                        help="Force re-tokenization even if cached tokenized data exists.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max number of training samples to use.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Max number of validation samples to use.")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Max number of test samples to use.") # ADDED
    
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
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Report metrics to.")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load base model in 4-bit (QLoRA).")
    parser.add_argument("--use_flash_attention_2", action="store_true", help="Use Flash Attention 2.")

    args = parser.parse_args()
    set_seed(args.seed)

    # --- 1. Load Tokenizer ---
    model_config_details = MODEL_CONFIGS[args.model_name]
    hf_model_name = model_config_details["hf_model_name"]
    add_eos_token_to_output = model_config_details.get("add_eos_token", True)

    print(f"Loading tokenizer for {hf_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, cache_dir=args.base_model_cache_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Tokenizer missing pad_token, setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 2. Dataset Preparation ---
    tokenized_train_dataset: Optional[Dataset] = None
    tokenized_eval_dataset: Optional[Dataset] = None
    tokenized_test_dataset: Optional[Dataset] = None # ADDED
    
    sanitized_model_name = hf_model_name.split('/')[-1].replace('-', '_')
    
    dataset_splits = {"train": args.train_file, "validation": args.val_file}
    if args.test_file:
        dataset_splits["test"] = args.test_file

    tokenized_datasets_loaded = {} # To store loaded/processed datasets

    # Loop for train, validation, and test
    for split_name, raw_file_path in dataset_splits.items():
        if not raw_file_path: continue # Skip if test_file is not provided

        cache_path_for_split = None
        if args.tokenized_data_path:
            # Each split (train, val, test) gets its own directory
            cache_dir_for_split = os.path.join(args.tokenized_data_path, f"{sanitized_model_name}_seq{args.max_seq_length}_{split_name}")
            cache_path_for_split = cache_dir_for_split # This path is a directory for save_to_disk/load_from_disk
            
            if not args.overwrite_tokenized_cache and os.path.exists(os.path.join(cache_path_for_split, "dataset_info.json")):
                print(f"Loading tokenized {split_name} data from cache: {cache_path_for_split}")
                try:
                    tokenized_datasets_loaded[split_name] = Dataset.load_from_disk(cache_path_for_split)
                    continue # Skip to next split if loaded from cache
                except Exception as e:
                    print(f"Failed to load {split_name} cache: {e}. Will re-tokenize.")
        
        # If not loaded from cache, tokenize
        print(f"Tokenizing {split_name} data from {raw_file_path}...")
        # Create a dict for load_dataset for a single split
        current_data_file = {split_name: raw_file_path}
        raw_dataset_split = load_dataset("json", data_files=current_data_file)[split_name]

        def preprocess_function(examples):
            inputs_text = [] 
            prompts_text = examples["input"]
            for i in range(len(prompts_text)):
                text = prompts_text[i] + examples["output"][i]
                if add_eos_token_to_output: text += tokenizer.eos_token
                inputs_text.append(text)
            model_inputs = tokenizer(inputs_text, max_length=args.max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
            labels = model_inputs["input_ids"].clone()
            for i in range(len(prompts_text)):
                prompt_tokens = tokenizer(prompts_text[i], max_length=args.max_seq_length, truncation=True, add_special_tokens=True)
                prompt_tokens_length = sum(torch.tensor(prompt_tokens['attention_mask'])).item()
                mask_len = min(prompt_tokens_length, labels.shape[1])
                labels[i, :mask_len] = -100
            model_inputs["labels"] = labels
            return model_inputs

        num_proc = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
        tokenized_split = raw_dataset_split.map(
            preprocess_function, batched=True, remove_columns=raw_dataset_split.column_names,
            num_proc=num_proc, desc=f"Tokenizing {split_name} set"
        )
        tokenized_datasets_loaded[split_name] = tokenized_split

        if cache_path_for_split: # Save if path was provided
            print(f"Saving tokenized {split_name} data to cache: {cache_path_for_split}")
            # save_to_disk expects a directory path where it will create its files
            os.makedirs(cache_path_for_split, exist_ok=True)
            tokenized_split.save_to_disk(cache_path_for_split)

    tokenized_train_dataset = tokenized_datasets_loaded.get("train")
    tokenized_eval_dataset = tokenized_datasets_loaded.get("validation")
    tokenized_test_dataset = tokenized_datasets_loaded.get("test") # Will be None if no test_file

    # --- Sub-sampling ---
    if args.max_train_samples is not None and args.max_train_samples > 0 and tokenized_train_dataset:
        if args.max_train_samples < len(tokenized_train_dataset):
            print(f"Sub-sampling training data to {args.max_train_samples} samples.")
            tokenized_train_dataset = tokenized_train_dataset.select(range(args.max_train_samples))
    if args.max_eval_samples is not None and args.max_eval_samples > 0 and tokenized_eval_dataset:
        if args.max_eval_samples < len(tokenized_eval_dataset):
            print(f"Sub-sampling validation data to {args.max_eval_samples} samples.")
            tokenized_eval_dataset = tokenized_eval_dataset.select(range(args.max_eval_samples))
    if args.max_test_samples is not None and args.max_test_samples > 0 and tokenized_test_dataset: # ADDED
        if args.max_test_samples < len(tokenized_test_dataset):
            print(f"Sub-sampling test data to {args.max_test_samples} samples.")
            tokenized_test_dataset = tokenized_test_dataset.select(range(args.max_test_samples))


    if not tokenized_train_dataset or not tokenized_eval_dataset:
        print("Error: Training or Validation dataset is empty or failed to load/process. Exiting.")
        sys.exit(1)

    print(f"Final train dataset size: {len(tokenized_train_dataset)}")
    print(f"Final validation dataset size: {len(tokenized_eval_dataset)}")
    if tokenized_test_dataset:
        print(f"Final test dataset size: {len(tokenized_test_dataset)}")


    # --- Tokenize-Only Mode Exit ---
    if args.tokenize_only:
        if args.tokenized_data_path: print(f"Tokenization complete. Processed datasets saved/cached in: {args.tokenized_data_path}")
        else: print("Tokenization complete. No --tokenized_data_path, data not persistently saved by script.")
        sys.exit(0)

    # --- If not tokenize_only, proceed to load full model and train ---
    # ... (Model loading, LoRA setup, TrainingArguments, Trainer, train as before) ...
    if not args.output_dir: parser.error("--output_dir required if not --tokenize_only.")
    target_modules = model_config_details["target_modules"]
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                             bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                                             bnb_4bit_use_double_quant=True)
    print(f"Loading base model for training: {hf_model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name, cache_dir=args.base_model_cache_dir, quantization_config=quantization_config,
        torch_dtype="auto", device_map="auto", trust_remote_code=True,
        attn_implementation="flash_attention_2" if args.use_flash_attention_2 and quantization_config is None else "sdpa",
    )
    if model.config.pad_token_id is None: model.config.pad_token_id = tokenizer.eos_token_id
    if args.load_in_4bit: model = prepare_model_for_kbit_training(model)
    print("Setting up LoRA configuration...")
    lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=target_modules, lora_dropout=args.lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config); model.print_trainable_parameters()
    print("Setting up Training Arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size, per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio, lr_scheduler_type=args.lr_scheduler_type, optim=args.optim,
        logging_dir=os.path.join(args.output_dir, "logs"), logging_steps=args.logging_steps,
        evaluation_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        # For loading best model, need to specify metric. Often "eval_loss"
        load_best_model_at_end=True if args.val_file and args.eval_steps > 0 else False, # Only if eval is done
        metric_for_best_model="eval_loss" if args.val_file and args.eval_steps > 0 else None,
        greater_is_better=False if args.val_file and args.eval_steps > 0 else None,
        report_to=args.report_to.split(',') if args.report_to and args.report_to != "none" else None,
        fp16=torch.cuda.is_available() and not args.load_in_4bit and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported() and not args.load_in_4bit, seed=args.seed,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    print("Initializing Trainer..."); trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_train_dataset, eval_dataset=tokenized_eval_dataset, tokenizer=tokenizer, data_collator=data_collator)
    print("Starting training..."); trainer.train(); print("Training complete.")
    
    # --- Evaluate on Test Set (if provided) ---
    if tokenized_test_dataset:
        print("\nEvaluating on the test set...")
        test_results = trainer.evaluate(eval_dataset=tokenized_test_dataset, metric_key_prefix="test")
        print(f"Test Set Metrics: {test_results}")
        # Save test results
        test_results_path = os.path.join(args.output_dir, "test_results.json")
        with open(test_results_path, "w") as f:
            json.dump(test_results, f, indent=4)
        print(f"Test results saved to {test_results_path}")

    # --- Save Final LoRA Adapter ---
    final_adapter_path = os.path.join(args.output_dir, "final_lora_adapter")
    print(f"Saving final LoRA adapter to {final_adapter_path}..."); model.save_pretrained(final_adapter_path)
    print("LoRA adapter saved successfully."); print(f"Base model was: {hf_model_name}")


if __name__ == "__main__":
    main()