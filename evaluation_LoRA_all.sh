#!/bin/bash

# Check if GPU count argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <number_of_gpus>"
  exit 1
fi

NUM_GPUS=$1

LORA_PATH=$2
TAG=$3

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=test.job
#SBATCH --output=/home/skuo/out_evaluation_LoRA_all_%j.txt
#SBATCH --error=/home/skuo/err_evaluation_LoRA_all_%j.txt
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

python evaluation.py \
    --model_name "TinyLLaMA" \
    --lora_adapter_path ${LORA_PATH} \
    --test_file training_data/phase1/test.jsonl \
    --explanation_test_folder training_data/phase2_corrected/test \
    --stockfish_path ../stockfish-11-linux/Linux/stockfish_20011801_x64 \
    --output_results_file ./evaluation_results/${TAG}/result.json \
    --output_numerical_summary ./evaluation_results/${TAG}/summary.txt \
    --inference_cache_folder ./evaluation_results/${TAG}_inference_cache \
    --base_model_cache_dir ./hf_cache \
    --max_p1_eval_samples 200 \
    --max_p2_eval_samples 200 \
    --eval_move_pred \
    --eval_rule_tasks \
    --eval_explanation \
    --stockfish_analysis_time 0.3 \
    --top_k_agreement 1 3 5 10 50 100 \
    --bert_score_model_type "microsoft/deberta-xlarge-mnli" \
    --max_seq_length 1024 \
    --batch_size 64 \
    --seed 42 \
    --default_max_new_tokens 150 \
    --load_in_4bit
EOT