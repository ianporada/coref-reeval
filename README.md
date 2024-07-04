# A Controlled Reevaluation of Coreference Resolution Models

## Models

See [models/](models/) for each respective models' training and inference code. All models are implemented in PyTorch, referencing the original implementations as described in the paper.

### Encoder models

The code for the four encoder models is at [models/encoder_based/](models/encoder_based/). Encoder-based models are trained using PyTorch Lightning.

### LinkAppend

Our implementation of the LinkAppend model is available at [models/decoder_based/LinkAppend/](models/decoder_based/LinkAppend/). In particular, for details on training and inference see the [README](models/decoder_based/LinkAppend/README.md). The LinkAppend model can be trained using HuggingFace's Trainer.

## Details

This repo includes the raw model code. Models implemented by @XiyuanZou and @ianporada.

For details, see [the paper](https://aclanthology.org/2024.lrec-main.23/):
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
