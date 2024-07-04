#!/bin/bash

module purge

# load environment
module load anaconda/3
conda activate coref_env

# set cache directory
mkdir -p $SCRATCH/cache
export TRANSFORMERS_CACHE="${SCRATCH}/cache/transformers_cache"
export HF_DATASETS_CACHE="${SCRATCH}/cache/datasets_cache"

#inference
cd ~/research/kd-coref
mkdir -p $SCRATCH/kd-coref-project-output
python inference/inference_lingmess.py output_dir=$SCRATCH/kd-coref-project-output cache_dir=$HF_DATASETS_CACHE

#list of experiments:
    #inference_lingmess
    #inference_s2e
    #inference_c2f