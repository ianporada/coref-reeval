#!/bin/bash

module load anaconda/3
conda activate coref_env

wandb login
huggingface-cli login
