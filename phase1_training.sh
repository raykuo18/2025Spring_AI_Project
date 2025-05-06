#!/bin/bash

# Check if GPU count argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <number_of_gpus>"
  exit 1
fi

NUM_GPUS=$1

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=test.job
#SBATCH --output=/home/skuo/out_phase1_training_%j.txt
#SBATCH --error=/home/skuo/er_phase1_training_%j.txt
#SBATCH --time=2-00:00
#SBATCH --mem=70000
#SBATCH --gres=gpu:${NUM_GPUS}

# Test CUDA compiler (not needed by deep learning people, we just use the python libraries)
# /cm/shared/apps/cuda11.1/toolkit/11.1.1/bin/nvcc -o saxpy /home/<netid>/cuda_c_code/saxpy.cu && ./saxpy

# Test nvidia-smi
nvidia-smi

# Test Python conda environment
# source activate your_conda_env
# /home/<netid>/miniconda3/envs/your_conda_env/bin/python /home/<netid>/run.py
cd /home/skuo/2025Spring_AI_Project

# Example for TinyLLaMA
python phase1_training.py \
    --model_name "TinyLLaMA" \
    --train_file training_data/phase1/train.jsonl \
    --val_file training_data/phase1/val.jsonl \
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
EOT
