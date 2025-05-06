#!/usr/bin/env bash

# Prior to job sumbission, run:
# 
# source activate cse537 # Do this on CLI prior to job submission
# 

# Submit job command: ./run-train_inference.sh

USER=$(whoami)

sbatch <<EOT
#!/usr/bin/env bash
#SBATCH --job-name=train-inf
#SBATCH --output=/home/${USER}/cse537/2025Spring_AI_Project/output-train_experiment-%j.txt
#SBATCH --error=/home/${USER}/cse537/2025Spring_AI_Project/error-train_experiment-%j.txt
#SBATCH --time=6-00:00
#SBATCH --mem=70000
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=${USER}@cs.stonybrook.edu

scripts_dir=/home/${USER}/cse537/2025Spring_AI_Project
cd \${scripts_dir}

# Run python script
python /home/adbraimah/cse537/2025Spring_AI_Project/training_inference_pipline_validate.py
EOT