#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --job-name=xenominer
#SBATCH --output=/home/bioeda/bio/XenoMiner/dataset/slurm_logs/xenominer_%j.log
#SBATCH --mem=16GB
#SBATCH --mail-user=marilisa.montemurro@polito.it
#SBATCH --mail-type=ALL
##
# load cuda module
module load nvidia/cudasdk


# Activate Anaconda work environment
source /home/${USER}/.bashrc
source activate xenominer

# Import path for CUDA libraries
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mmontemurro/miniconda3/envs/xenominer/lib
prj_root="/home/bioeda/bio/XenoMiner"

python ${prj_root}/local/src/models/CNN.py -i ${prj_root}/dataset/datasets/chr1_150_12mer -o ${prj_root}/dataset/results/chr1_150_12mer_gpu_
