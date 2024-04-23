# About

This directory contains our reimplementation of the LinkAppend model from the paper "Coreference Resolution through a seq2seq Transition-Based System" by Bohnet et al.

## Inference

### Setup

First create the environment using conda (or the local requirements.txt):
```bash
conda create -y -n coref python=3.11 pytorch torchvision torchaudio pytorch-cuda=12.1
conda activate coref
conda install -y transformers=4.37.2 datasets wandb hydra-core sentencepiece
```

Then download the model weights from ['mt5-coref-pytorch/link-append-xxl'](https://huggingface.co/mt5-coref-pytorch/link-append-xxl). These were converted from the released t5x checkpoint to HuggingFace using [convert_t5x_checkpoint_to_flax.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/convert_t5x_checkpoint_to_flax.py).

### Inference on a dataset

For example, to run inference on the first 10 documents of the PreCo test set run (inference fits on a single A100 80GB GPU):
```bash
conda activate coref

python main.py \
    --model_path $MODEL_CHECKPOINT \
    --max_input_size 3000 \
    --output_dir $OUTPUT \
    --split test \
    --batch_size 4 \
    --dataset_name preco \
    --no_pound_symbol \
    --subset 10 \
    --subset_start 0
```


## References

For details on our reimplementation of LinkAppend, see the paper:
```
@misc{porada2024controlled,
    title={A Controlled Reevaluation of Coreference Resolution Models},
    author={Ian Porada and Xiyuan Zou and Jackie Chi Kit Cheung},
    year={2024},
    eprint={2404.00727},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

Please also see the original paper from Bohnet et al.:
```
@article{bohnet-etal-2023-coreference,
    title = "Coreference Resolution through a seq2seq Transition-Based System",
    author = "Bohnet, Bernd  and
      Alberti, Chris  and
      Collins, Michael",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "11",
    year = "2023",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2023.tacl-1.13",
    doi = "10.1162/tacl_a_00543",
    pages = "212--226",
    abstract = "Most recent coreference resolution systems use search algorithms over possible spans to identify mentions and resolve coreference. We instead present a coreference resolution system that uses a text-to-text (seq2seq) paradigm to predict mentions and links jointly. We implement the coreference system as a transition system and use multilingual T5 as an underlying language model. We obtain state-of-the-art accuracy on the CoNLL-2012 datasets with 83.3 F1-score for English (a 2.3 higher F1-score than previous work [Dobrovolskii, 2021]) using only CoNLL data for training, 68.5 F1-score for Arabic (+4.1 higher than previous work), and 74.3 F1-score for Chinese (+5.3). In addition we use the SemEval-2010 data sets for experiments in the zero-shot setting, a few-shot setting, and supervised setting using all available training data. We obtain substantially higher zero-shot F1-scores for 3 out of 4 languages than previous approaches and significantly exceed previous supervised state-of-the-art results for all five tested languages. We provide the code and models as open source.1",
}
```