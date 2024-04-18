#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -J fcn
#SBATCH -o sfno_forward.out

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

srun python train.py --enable_amp --yaml_config=./config/AFNO.yaml --config=$config --run_num=$run_num
