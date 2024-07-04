# About

This directory contains our reimplementation of the LinkAppend model from the paper "Coreference Resolution through a seq2seq Transition-Based System" by Bohnet et al.

## Inference

### Setup

First create the environment using conda (or the local requirements.txt):
```bash
conda create -y -n linkappend_env python=3.11 pytorch torchvision torchaudio pytorch-cuda=12.1
conda activate linkappend_env
conda install -y transformers=4.37.2 datasets wandb hydra-core sentencepiece
```

Then download the model weights from ['mt5-coref-pytorch/link-append-xxl'](https://huggingface.co/mt5-coref-pytorch/link-append-xxl). These were converted from the released t5x checkpoint to HuggingFace using [convert_t5x_checkpoint_to_flax.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/convert_t5x_checkpoint_to_flax.py).

### Inference on a dataset

For example, to run inference on the first 10 documents of the PreCo test set run (inference fits on a single A100 80GB GPU):
```bash
conda activate linkappend_env

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

### Finetuning

Set `--model_path` to `'oracle'` and inference will generate the input/output examples for an oracle model in the output directory. Then finetune a seq2seq model on these input/output pairs. See `finetuning/main_large.py` for an example finetuning mT5-large.

## References

For details on our reimplementation of LinkAppend, see the paper:
```
@inproceedings{porada-etal-2024-controlled-reevaluation,
    title = "A Controlled Reevaluation of Coreference Resolution Models",
    author = "Porada, Ian  and
      Zou, Xiyuan  and
      Cheung, Jackie Chi Kit",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.23",
    pages = "256--263",
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
}
```