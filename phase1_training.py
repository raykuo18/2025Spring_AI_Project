#!/usr/bin/env python3

import argparse
import os
import json
import random
import numpy as np
import torch
import sys
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer, # We will subclass this
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
# tqdm is implicitly used by datasets.map and Trainer

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

# --- Custom Trainer for Robust Loss Computation on Multi-GPU ---
class CustomSafeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Ensures labels are on the same device as model outputs (logits) before loss calculation,
        especially relevant for models sharded with device_map="auto".
        """
        # If model is a PeftModel, model.device gives the device of the first parameter,
        # which is usually where inputs should initially go.
        # Logits might be on a different device if lm_head is sharded.

        # Get labels and remove from inputs if you plan to pass them separately
        # to a manual loss function. However, Hugging Face models usually
        # compute loss internally if 'labels' are in **inputs.
        labels = inputs.get("labels")

        # Ensure labels are on the correct device *before* passing to model if model computes loss internally
        # Or before using them with logits if computing loss manually.
        # We will let the model compute the loss by passing labels.
        # The model's forward pass with device_map should handle internal tensor movements.
        # The critical point is that the CrossEntropyLoss (or equivalent) inside the model's
        # forward method receives logits and labels on the same device.

        if labels is not None:
            # Determine target device for labels. This should be the device of the logits.
            # We can't know logits' device before the forward pass.
            # So, we pass labels as is. If the model's internal loss fails due to device mismatch,
            # it indicates a deeper issue with accelerate/peft.
            # However, sometimes, an explicit move in a custom trainer is a workaround.
            # Let's try passing inputs as is first, and if error persists, consider this:
            # dummy_outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            # if hasattr(dummy_outputs, "logits"):
            #     inputs["labels"] = labels.to(dummy_outputs.logits.device)
            pass # Usually, Trainer handles moving inputs to model.device

        outputs = model(**inputs)

        if labels is not None and hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        elif labels is not None and hasattr(outputs, "logits"):
            # Manually compute loss if model didn't return it (e.g., if labels were popped)
            # This part assumes labels were NOT popped before model(**inputs)
            # and the model's forward just didn't return a loss attribute directly
            # but did return logits.
            logits = outputs.logits
            # Ensure labels are on the same device as logits
            labels_on_device = labels.to(logits.device)
            
            loss_fct = torch.nn.CrossEntropyLoss()
            # Standard Causal LM shifting for labels and logits
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_on_device[..., 1:].contiguous()
            # Flatten the tokens
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            # This case should not happen in standard CausalLM training with labels
            loss = None
            if not hasattr(outputs, "loss") or outputs.loss is None :
                 print("Warning: Model did not return loss and manual computation prerequisites not met.")


        return (loss, outputs) if return_outputs else loss

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
    parser.add_argument("--tokenized_data_path", type=str, default=None, help="Base path to save/load tokenized datasets.")
    parser.add_argument("--overwrite_tokenized_cache", action="store_true", help="Force re-tokenization.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples to load/tokenize.")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max validation samples to load/tokenize.") # Corrected name
    parser.add_argument("--max_test_samples", type=int, default=None, help="Max test samples to load/tokenize.")
    parser.add_argument("--tokenize_only", action="store_true", help="If set, load data, tokenize, save tokenized data, and exit.")
    
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
    # --- ADDED/MODIFIED for gradient checkpointing warning ---
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--gradient_checkpointing_use_reentrant", type=bool, default=False, help="Value for use_reentrant in gradient_checkpointing_kwargs (False is recommended).")


    # Logging and Saving
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X steps.")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every X steps.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit total checkpoints.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Report metrics to (e.g. tensorboard, wandb, none).")

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
    tokenized_test_dataset: Optional[Dataset] = None
    
    sanitized_model_name = hf_model_name.split('/')[-1].replace('-', '_')
    
    datasets_to_process_config = {
        "train": {"file": args.train_file, "max_samples": args.max_train_samples, "cache_suffix": "train"},
        "validation": {"file": args.val_file, "max_samples": args.max_val_samples, "cache_suffix": "eval"}, # Used corrected arg name
    }
    if args.test_file:
        datasets_to_process_config["test"] = {"file": args.test_file, "max_samples": args.max_test_samples, "cache_suffix": "test"}

    tokenized_datasets_loaded = {}

    for split_name, config in datasets_to_process_config.items():
        raw_file_path = config["file"]
        max_samples_for_split = config["max_samples"]
        cache_suffix_for_split = config["cache_suffix"]
        cache_path_for_split = None

        if args.tokenized_data_path:
            subset_info_suffix = f"_top{max_samples_for_split}" if max_samples_for_split is not None else ""
            cache_dir_for_split = os.path.join(args.tokenized_data_path, f"{sanitized_model_name}_seq{args.max_seq_length}_{cache_suffix_for_split}{subset_info_suffix}")
            cache_path_for_split = cache_dir_for_split
            
            if not args.overwrite_tokenized_cache and os.path.exists(os.path.join(cache_path_for_split, "dataset_info.json")):
                print(f"Loading tokenized {split_name} data from cache: {cache_path_for_split}")
                try: tokenized_datasets_loaded[split_name] = Dataset.load_from_disk(cache_path_for_split); print(f"Loaded {len(tokenized_datasets_loaded[split_name])} samples for {split_name} from cache."); continue
                except Exception as e: print(f"Failed to load {split_name} cache: {e}. Will re-tokenize.")
        
        print(f"Preparing to tokenize {split_name} data from {raw_file_path}...")
        current_data_file_dict_for_load = {split_name: raw_file_path}
        slicing_instruction = split_name
        if max_samples_for_split is not None and max_samples_for_split > 0:
            slicing_instruction = f"{split_name}[:{max_samples_for_split}]"
            print(f"Will load a maximum of {max_samples_for_split} raw samples for {split_name} set using split='{slicing_instruction}'.")
        try: raw_dataset_this_split = load_dataset("json", data_files=current_data_file_dict_for_load, split=slicing_instruction)
        except Exception as e: print(f"Error loading raw {split_name} data: {e}. Skipping this split."); continue

        def preprocess_function(examples):
            inputs_text = []; prompts_text = examples["input"]; outputs_text = examples["output"]
            for i in range(len(prompts_text)):
                output_str = str(outputs_text[i]) if outputs_text[i] is not None else ""
                text = prompts_text[i] + output_str 
                if add_eos_token_to_output: text += tokenizer.eos_token
                inputs_text.append(text)
            model_inputs = tokenizer(inputs_text, max_length=args.max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
            labels = model_inputs["input_ids"].clone()
            for i in range(len(prompts_text)):
                prompt_tokens = tokenizer(prompts_text[i], max_length=args.max_seq_length, truncation=True, add_special_tokens=True) 
                prompt_tokens_length = sum(torch.tensor(prompt_tokens['attention_mask'])).item()
                mask_len = min(prompt_tokens_length, labels.shape[1]); labels[i, :mask_len] = -100
            model_inputs["labels"] = labels; return model_inputs

        num_proc = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
        print(f"Tokenizing {split_name} set ({len(raw_dataset_this_split)} raw samples) with {num_proc} processes...")
        tokenized_split = raw_dataset_this_split.map(preprocess_function, batched=True, remove_columns=raw_dataset_this_split.column_names, num_proc=num_proc, desc=f"Tokenizing {split_name} set")
        tokenized_datasets_loaded[split_name] = tokenized_split
        if cache_path_for_split:
            print(f"Saving tokenized {split_name} data to cache: {cache_path_for_split}")
            os.makedirs(cache_path_for_split, exist_ok=True); tokenized_split.save_to_disk(cache_path_for_split)

    tokenized_train_dataset = tokenized_datasets_loaded.get("train")
    tokenized_eval_dataset = tokenized_datasets_loaded.get("validation")
    tokenized_test_dataset = tokenized_datasets_loaded.get("test")

    if not tokenized_train_dataset: print("Error: Training dataset missing. Exiting."); sys.exit(1)
    # Val dataset is optional for trainer, but good to have if val_file was specified
    if args.val_file and not tokenized_eval_dataset: print("Warning: Validation dataset specified but could not be prepared.")
    if args.test_file and not tokenized_test_dataset: print("Warning: Test dataset specified but could not be prepared.")

    print(f"Final train dataset size: {len(tokenized_train_dataset) if tokenized_train_dataset else 0}")
    print(f"Final validation dataset size: {len(tokenized_eval_dataset) if tokenized_eval_dataset else 0}")
    if tokenized_test_dataset: print(f"Final test dataset size: {len(tokenized_test_dataset)}")

    if args.tokenize_only:
        if args.tokenized_data_path: print(f"Tokenization complete. Datasets saved in: {args.tokenized_data_path}")
        else: print("Tokenization complete. No --tokenized_data_path, data not saved by script.")
        sys.exit(0)

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
    
    # --- Set use_cache=False for training ---
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        print(f"Setting model.config.use_cache to False for training. Original: {model.config.use_cache}")
        model.config.use_cache = False

    if args.load_in_4bit: 
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    # --- Enable gradient checkpointing if specified (and if not using QLoRA's implicit version) ---
    # `prepare_model_for_kbit_training` might handle it for QLoRA.
    # If not using QLoRA but want gradient checkpointing:
    if args.gradient_checkpointing and not args.load_in_4bit:
        print("Enabling gradient checkpointing for non-QLoRA model.")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant})
    elif args.gradient_checkpointing and args.load_in_4bit:
        # prepare_model_for_kbit_training already took use_gradient_checkpointing.
        # If we still want to explicitly pass use_reentrant for the QLoRA case, we might need to ensure it.
        # For now, rely on prepare_model_for_kbit_training.
        print("Gradient checkpointing handled by prepare_model_for_kbit_training for QLoRA.")


    print("Setting up LoRA configuration..."); lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=target_modules, lora_dropout=args.lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config); model.print_trainable_parameters()

    # Re-check use_cache after PEFT, as get_peft_model might re-enable it
    if hasattr(model, "config") and hasattr(model.config, "use_cache") and model.config.use_cache:
        print(f"Re-setting model.config.use_cache to False after PEFT model wrapping.")
        model.config.use_cache = False


    print("Setting up Training Arguments..."); 
    training_args_dict = {
        "output_dir":args.output_dir, "num_train_epochs":args.num_train_epochs,
        "per_device_train_batch_size":args.per_device_train_batch_size, 
        "per_device_eval_batch_size":args.per_device_eval_batch_size if tokenized_eval_dataset else args.per_device_train_batch_size,
        "gradient_accumulation_steps":args.gradient_accumulation_steps, "learning_rate":args.learning_rate,
        "weight_decay":args.weight_decay, "warmup_ratio":args.warmup_ratio, "lr_scheduler_type":args.lr_scheduler_type, 
        "optim":args.optim, "logging_dir":os.path.join(args.output_dir, "logs"), "logging_steps":args.logging_steps,
        "evaluation_strategy":"steps" if tokenized_eval_dataset and args.eval_steps > 0 else "no", 
        "eval_steps":args.eval_steps if tokenized_eval_dataset and args.eval_steps > 0 else None,
        "save_strategy":"steps", "save_steps":args.save_steps, "save_total_limit":args.save_total_limit,
        "load_best_model_at_end":True if tokenized_eval_dataset and args.eval_steps > 0 else False,
        "metric_for_best_model":"eval_loss" if tokenized_eval_dataset and args.eval_steps > 0 else None,
        "greater_is_better":False if tokenized_eval_dataset and args.eval_steps > 0 else None,
        "report_to":args.report_to.split(',') if args.report_to and args.report_to != "none" else None,
        "fp16":torch.cuda.is_available() and not args.load_in_4bit and not torch.cuda.is_bf16_supported(),
        "bf16":torch.cuda.is_bf16_supported() and not args.load_in_4bit, "seed":args.seed,
    }
    if args.gradient_checkpointing: # Pass gradient checkpointing args
        training_args_dict["gradient_checkpointing"] = True
        training_args_dict["gradient_checkpointing_kwargs"] = {"use_reentrant": args.gradient_checkpointing_use_reentrant}
    
    training_args = TrainingArguments(**training_args_dict)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    print("Initializing CustomSafeTrainer...")
    trainer = CustomSafeTrainer(model=model, args=training_args, train_dataset=tokenized_train_dataset, eval_dataset=tokenized_eval_dataset, tokenizer=tokenizer, data_collator=data_collator)
    
    print("Starting training..."); trainer.train(); print("Training complete.")
    
    if tokenized_test_dataset:
        print("\nEvaluating on the test set..."); test_results = trainer.evaluate(eval_dataset=tokenized_test_dataset, metric_key_prefix="test")
        print(f"Test Set Metrics: {test_results}")
        test_results_path = os.path.join(args.output_dir, "test_results.json")
        with open(test_results_path, "w") as f: json.dump(test_results, f, indent=4)
        print(f"Test results saved to {test_results_path}")
    
    final_adapter_path = os.path.join(args.output_dir, "final_lora_adapter")
    print(f"Saving final LoRA adapter to {final_adapter_path}..."); model.save_pretrained(final_adapter_path)
    print("LoRA adapter saved successfully."); print(f"Base model was: {hf_model_name}")

if __name__ == "__main__":
    main()