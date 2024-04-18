#!/bin/bash -l
#SBATCH --time=6:00:00
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=110G
#SBATCH -J fcn
#SBATCH -o compute_mse.out

. /usr/share/modules/init/bash
module purge
module load anaconda/py3.9 nvidia/11.6 git/2.38.1
conda activate fcn
cd /discover/nobackup/awang17/FourCastNet

export MASTER_ADDR=$(hostname)

config_file=./config/AFNO.yaml
config='afno_backbone_finetune'
run_num='2'

wandb disabled
echo $config

srun python inference/compute_mse.py --config=$config --run_num=$run_num
