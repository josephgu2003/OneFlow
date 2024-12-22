#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --partition=jiang
#SBATCH --gres=gpu:a5000:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=24GB

# Commands to execute
module load anaconda3/2022.05
module load gcc/10.1.0
module load cuda/12.1


source activate ../threestudio/threestudio_venv

data_dir=/work/vig/Datasets/FlyingChairs_release/data

log_dir=./log/dino_896_PRETRAINED_FIXED_ATTN/cont2

model=Dino
model_cfg=None
resume=None

mkdir $log_dir

echo $log_dir
git diff HEAD > $log_dir/git_diff
git status > $log_dir/git_status

python setup.py install

PYTHONPATH=/home/gu.jo/.cache/torch/hub/facebookresearch_dinov2_main

python -u -m tools.train --train_cfg configs/trainers/dit/dit_chairs_baseline.yaml --train_ds 'FlyingChairs' \
--train_data_dir ${data_dir} --val_ds 'FlyingChairs' --val_data_dir ${data_dir} --model $model \
--log_dir ${log_dir} --ckpt_dir ${log_dir} --device=0,1 --custom_cfg $model_cfg --resume $resume
