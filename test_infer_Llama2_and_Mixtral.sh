#!/bin/bash

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=test.job
#SBATCH --output=/home/skuo/test_out.txt
#SBATCH --error=/home/skuo/test_err.txt
#SBATCH --time=2-00:00
#SBATCH --mem=50000
#SBATCH --gres=gpu:2

# Test CUDA compiler (not needed by deep learning people, we just use the python libraries)
/cm/shared/apps/cuda11.1/toolkit/11.1.1/bin/nvcc -o saxpy /home/<netid>/cuda_c_code/saxpy.cu && ./saxpy

# Test nvidia-smi
nvidia-smi

# Test Python conda environment
# source activate your_conda_env
# /home/<netid>/miniconda3/envs/your_conda_env/bin/python /home/<netid>/run.py
python3 test_infer_Llama2_and_Mixtral.py
EOT
