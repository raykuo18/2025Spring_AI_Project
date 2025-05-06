# Example for TinyLLaMA
python your_training_script_name.py \
    --model_name "TinyLLaMA" \
    --train_file ./path_to/train.jsonl \
    --val_file ./path_to/val.jsonl \
    --output_dir ./output_adapters/tinyllama_phase1_lora \
    --base_model_cache_dir ./hf_cache \
    --max_seq_length 1024 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --save_total_limit 2 \
    --load_in_4bit # Add this if you want QLoRA
    # --use_flash_attention_2 # Add if your setup supports it and not using 4-bit