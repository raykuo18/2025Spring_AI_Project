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
#SBATCH --output=/home/skuo/out_evaluation_LoRA_p2_%j.txt
#SBATCH --error=/home/skuo/err_evaluation_LoRA_p2_%j.txt
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
    --lora_adapter_path training_output/tinyllama_phase2_from_LoRA_checkpoint-10000/TinyLLaMA_phase2_20250508_08140932/phase2_explainer_lora/phase2_explainer_lora/ \
    --test_file training_data/phase1/test.jsonl \
    --explanation_test_folder training_data/phase2_corrected/test \
    --stockfish_path ../stockfish-11-linux/Linux/stockfish_20011801_x64 \
    --output_results_file ./evaluation_results/phase2_explainer_lora/result.json \
    --output_numerical_summary ./evaluation_results/phase2_explainer_lora/summary.txt \
    --inference_cache_folder ./evaluation_results/inference_cache \
    --base_model_cache_dir ./hf_cache \
    --max_p1_eval_samples 10000 \
    --max_p2_eval_samples 1000 \
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