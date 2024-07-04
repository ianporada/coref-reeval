#!/bin/bash

module purge

# load environment
module load anaconda/3
conda activate coref_env

# set cache directory
mkdir -p $SCRATCH/cache
export TRANSFORMERS_CACHE="${SCRATCH}/cache/transformers_cache"
export HF_DATASETS_CACHE="${SCRATCH}/cache/datasets_cache"

# train
cd ~/research/kd-coref
mkdir -p $SCRATCH/kd-coref-project-output
python experiments/train_s2e/train.py output_dir=$SCRATCH/kd-coref-project-output cache_dir=$HF_DATASETS_CACHE

# list of experiments:
    # train_s2e
    # train_lingmess_teacher
    # train_c2f