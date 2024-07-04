#!/bin/bash

module purge

module load anaconda/3
conda create -y -n coref_env python=3.9 cudatoolkit=11.3.1 --override-channels -c conda-forge -c nvidia
conda activate coref_env
conda install -y pytorch=1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --override-channels -c nvidia -c pytorch -c conda-forge -c defaults

conda install -y scikit-learn pandas matplotlib hydra-core --override-channels -c conda-forge
conda install -y transformers==4.27.2 datasets==2.10.1 sentencepiece pytorch-lightning==2.0.0 --override-channels -c conda-forge
conda install -y jsonlines wandb --override-channels -c conda-forge

pip install udapi
pip install nltk
