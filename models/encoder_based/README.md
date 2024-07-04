# Large Language Models for Coreference Resolution

[Shared Project Doc](https://docs.google.com/document/d/1QjnS299fbQINvgschjJIghNlDBqhQkAuXNpNcsbZu0g/edit#heading=h.gi489dvisd8f)

## Setup

Create the conda env on a login node (named `coref_env`):
```bash
./scripts/setup/setup_env.sh
```

Login to huggingface and wandb:
```bash
./scripts/setup/login.sh
```

## Run on the mila cluster

Sync the code to the cluster, e.g.
```bash
rsync -v -r -a --delete ~/Documents/llm-coref mila:~/research/
```

Request an interactive node:
```bash
salloc --gres=gpu:a100:1 -c 6 --mem=10G -t 1:00:00 --partition=unkillable
```

Train:
```bash
./scripts/train.sh
```

## Models

### Mention Ranking

* BERT4Coref https://github.com/mandarjoshi90/coref
  * Based on e2e-coref https://github.com/kentonl/e2e-coref
* wl-coref https://github.com/vdobrovolskii/wl-coref
* s2e-coref https://github.com/yuvalkirstain/s2e-coref

## Scorers

* CorefUD Scorer https://github.com/ufal/corefud-scorer
  * Based on Universal Anaphora Scorer https://github.com/juntaoy/universal-anaphora-scorer

## Data Preprocessing

OntoNotes 5.0 is hosted on LDC and the CoNLL 2012 Shared Task splits are hosted by Cemantix on [their website](https://cemantix.org/data/ontonotes.html) (English/Arabic/Chinese v4 and English v9) and [GitHub](https://github.com/ontonotes/conll-formatted-ontonotes-5.0) (English v12). These must be combined with the OntoNotes 5.0 data using the skeleton2conll.sh script. Someone already uploaded the combined data at [this Huggingface dataset](https://huggingface.co/datasets/conll2012_ontonotesv5). Coreference is stored in coref_spans (cluster_id, (start_index, end_index)) inclusive.
