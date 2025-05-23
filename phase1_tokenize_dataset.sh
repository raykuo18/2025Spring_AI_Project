python phase1_training.py \
    --tokenize_only \
    --model_name "TinyLLaMA" \
    --train_file training_data/phase1/train.jsonl \
    --val_file training_data/phase1/val.jsonl \
    --test_file training_data/phase1/test.jsonl \
    --tokenized_data_path training_data/phase1 \
    --output_dir ./training_output/tinyllama_phase1 \
    --base_model_cache_dir ./hf_cache \
    --max_seq_length 1024 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --optim "paged_adamw_8bit" \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --logging_steps 10 \
    --save_steps 200 \
    --eval_steps 100 \
    --save_total_limit 3 \
    --report_to "tensorboard" \
    --seed 42 \
    --load_in_4bit \
    --max_train_samples 600000 \
    --max_eval_samples 200000 \
    --max_test_samples 200000

    # --use_flash_attention_2 # Uncomment if your setup supports it AND you are NOT using --load_in_4bit (usually one or the other)
    # --max_train_samples 10000 # Optional: If you want to train on a subset of your tokenized train data
    # --max_eval_samples 1000   # Optional: If you want to evaluate on a subset of your tokenized val data
