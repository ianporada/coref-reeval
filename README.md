# A Controlled Reevaluation of Coreference Resolution Models

## Models

### Encoder models

The code fo the enocder models is at models/enocder_based

### LinkAppend

See the LinkAppend [README](models/decoder_based/LinkAppend/README.md).

## Details

All models are implemented in PyTorch, referencing the original implementations as described in the paper. Encoder-based models are trained using PyTorch Lightning,  and the decoder-based LinkAppend model is trained using HuggingFace's Trainer.

This repo includes the raw model code. Models implemented by @XiyuanZou and @ianporada.

For details, see [the paper](https://aclanthology.org/2024.lrec-main.23/):
```
@inproceedings{porada-etal-2024-controlled-reevaluation,
    title = "A Controlled Reevaluation of Coreference Resolution Models",
    author = "Porada, Ian  and
      Zou, Xiyuan  and
      Cheung, Jackie Chi Kit",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.23",
    pages = "256--263",
    abstract = "All state-of-the-art coreference resolution (CR) models involve finetuning a pretrained language model. Whether the superior performance of one CR model over another is due to the choice of language model or other factors, such as the task-specific architecture, is difficult or impossible to determine due to lack of a standardized experimental setup. To resolve this ambiguity, we systematically evaluate five CR models and control for certain design decisions including the pretrained language model used by each. When controlling for language model size, encoder-based CR models outperform more recent decoder-based models in terms of both accuracy and inference speed. Surprisingly, among encoder-based CR models, more recent models are not always more accurate, and the oldest CR model that we test generalizes the best to out-of-domain textual genres. We conclude that controlling for the choice of language model reduces most, but not all, of the increase in F1 score reported in the past five years.",
}
```
