# A Controlled Reevaluation of Coreference Resolution Models

## Models

### Encoder models

The code fo the enocder models is at models/enocder_based

### LinkAppend

See the LinkAppend [README](models/decoder_based/LinkAppend/README.md).

## Details

All models are implemented in PyTorch, referencing the original implementations as described in the paper. Encoder-based models are trained using PyTorch Lightning,  and the decoder-based LinkAppend model is trained using HuggingFace's Trainer.

This repo includes the raw model code. Models implemented by @XiyuanZou and @ianporada.

For details, see the paper:
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
