import os
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
import random, sys

from pytorch_lightning.utilities.types import EVAL_DATALOADERS
sys.path.append("src")
from utils import flatten, SPEAKER_START, SPEAKER_END, NULL_ID_FOR_COREF

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset, load_dataset
from pytorch_lightning import _logger as log
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, default_data_collator
import math
from typing import List, Iterable

class S2ECorefDataset(Dataset):
    def __init__(self, dataset_raw: Dataset, tokenizer, config): #config={"max_seq_len":int}
        self.dataset_raw=dataset_raw
        self.tokenizer=tokenizer
        self.max_seq_len=config["max_seq_len"]
        self.max_mention_num=-1
        self.max_cluster_size=-1
        self.max_num_clusters=-1
        self.num_examples_filtered=0
        self.coref_examples=[] #list of the input of each doc to the model
        
        for doc in self.dataset_raw:
            input_words_list=[]
            speakers_list=[]
            clusters_dict={}
            word_acc_idx=0
            for sent_dict in doc["sentences"]:
                input_words_list.append(sent_dict["words"])
                speakers_list.append([str(sent_dict["speaker"])]*len(sent_dict["words"]))
                for span in sent_dict["coref_spans"]:
                    if span[0] in clusters_dict:
                        clusters_dict[span[0]].append((span[1]+word_acc_idx, span[2]+word_acc_idx))
                    else:
                        clusters_dict[span[0]]=[(span[1]+word_acc_idx, span[2]+word_acc_idx)]
                word_acc_idx+=len(sent_dict["words"])
            input_words_list=flatten(input_words_list) #input_words_list(num_words): list of all words in the doc
            speakers_list=flatten(speakers_list) #speakers_list(num_words): list of speakers for each word
            clusters_list=[cluster for cluster_id, cluster in clusters_dict.items()] #clusters_list(num_clusters), each span is in the form (word_start_idx, word_end_idx), eg.[[(0,2),(4,5)],[(3,3),(6,9),(15,15)]]
            self.max_mention_num=max(self.max_mention_num, len(flatten(clusters_list)))
            self.max_cluster_size=max(self.max_cluster_size, max(len(cluster) for cluster in clusters_list) if clusters_list else 0)
            self.max_num_clusters=max(self.max_num_clusters, len(clusters_list) if clusters_list else 0)  
            
            word_idx_to_start_token_idx=dict()
            word_idx_to_end_token_idx=dict()
            end_token_idx_to_word_idx=[0]  # for <s>
            token_ids_list=[] #token_ids_list(num_subtokens): list of all subtokens in the doc
            last_speaker=None
            for idx, (word, speaker) in enumerate(zip(input_words_list, speakers_list)):
                if last_speaker!=speaker:
                    speaker_prefix=[SPEAKER_START]+self.tokenizer.encode(" " +speaker, add_special_tokens=False)+[SPEAKER_END]
                    last_speaker=speaker
                else:
                    speaker_prefix=[]
                for _ in range(len(speaker_prefix)):
                    end_token_idx_to_word_idx.append(idx)
                token_ids_list.extend(speaker_prefix)
                word_idx_to_start_token_idx[idx]=len(token_ids_list)+1  # +1 for <s>
                tokenized=self.tokenizer.encode(" " + word, add_special_tokens=False)
                for _ in range(len(tokenized)):
                    end_token_idx_to_word_idx.append(idx)
                token_ids_list.extend(tokenized)
                word_idx_to_end_token_idx[idx]=len(token_ids_list)
            #new_clusters(num_clusters), each span is in the form (subtoken_start_idx, subtoken_end_idx)
            new_clusters=[[(word_idx_to_start_token_idx[start], word_idx_to_end_token_idx[end]) for start, end in cluster] for cluster in clusters_list]
            
            if 0<self.max_seq_len<len(token_ids_list): #filter out all docs longer than the max sequence length allowed 
                self.num_examples_filtered+=1
                continue
            
            self.coref_examples.append((token_ids_list, new_clusters))
    
    def __len__(self):
        return len(self.coref_examples)

    def __getitem__(self, index):
        return self.coref_examples[index]
    
    def pad_clusters_inside(self, clusters):
        return [cluster+[(NULL_ID_FOR_COREF, NULL_ID_FOR_COREF)]*(self.max_cluster_size-len(cluster)) for cluster in clusters]

    def pad_clusters_outside(self, clusters):
        return clusters+[[]]*(self.max_num_clusters-len(clusters))

    def pad_clusters(self, clusters):
        clusters=self.pad_clusters_outside(clusters)
        clusters=self.pad_clusters_inside(clusters)
        return clusters

    def pad_batch(self, batch, max_length):
        max_length+=2  # we have additional two special tokens <s>, </s>
        input_ids=[]
        input_masks=[]
        gold_clusters=[]
        for example in batch:
            input_ids_per_example=[self.tokenizer.cls_token_id]+example[0]+[self.tokenizer.sep_token_id]+[self.tokenizer.pad_token_id]*(max_length-len(example[0])-2)
            input_masks_per_example=[1]*(len(example[0])+2)+[0]*(max_length-len(example[0])-2)
            gold_clusters_per_example=self.pad_clusters(example[1])
            input_ids.append(input_ids_per_example)
            input_masks.append(input_masks_per_example)
            gold_clusters.append(gold_clusters_per_example)
        input_ids=torch.tensor(input_ids, dtype=torch.long)
        input_masks=torch.tensor(input_masks, dtype=torch.long)
        gold_clusters=torch.tensor(gold_clusters, dtype=torch.long)
        return (input_ids, input_masks, gold_clusters)
    

class S2ECorefDataloader(DataLoader):
    def __init__(self, dataset: Dataset, num_workers, config): #config={"max_total_seq_len":int, "is_training":bool}
        dataset.coref_examples.sort(key=lambda x: len(x[0]), reverse=True)
        self.dataset=dataset
        self.max_total_seq_len=config["max_total_seq_len"]
        self.is_training=config["is_training"]
        self.num_workers=num_workers
        if self.is_training:
            self.batches=self.prepare_training_batches()
        else:
            self.batches=self.prepare_eval_batches() #when evaluating model, the batch size is 1
            
    def calc_effective_per_example_batch_len(self, example_len):
        return math.ceil((example_len+2)/512)*512
    
    def prepare_training_batches(self):
        batches=[]
        batch=[]
        per_example_batch_len=0
        for example in self.dataset.coref_examples:
            if len(batch)==0:
                per_example_batch_len=self.calc_effective_per_example_batch_len(len(example[0]))
            elif (len(batch)+1)*per_example_batch_len>self.max_total_seq_len:
                batch=self.dataset.pad_batch(batch, len(batch[0][0]))
                batches.append(batch)
                batch=[]
                per_example_batch_len=self.calc_effective_per_example_batch_len(len(example[0]))
            batch.append(example)
        if len(batch)==0:
            return batches
        batch=self.dataset.pad_batch(batch, len(batch[0][0]))
        batches.append(batch)
        return batches
    
    def prepare_eval_batches(self):
        batches=[]
        for example in self.dataset.coref_examples:
            batch=self.dataset.pad_batch([example], len(example[0]))
            batches.append(batch)
        return batches
        
    def __iter__(self) -> Iterable[List[int]]:
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
    
class S2ECorefDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int,
        eval_batch_size: int,
        test_batch_size: int,
        num_workers: int,
        max_length: int = 512,
        padding: Union[str, bool] = 'max_length',
        truncation: str = 'only_first',
        cache_dir: Optional[Union[Path, str]] = None,
        load_from_cache: bool = True,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.num_workers = num_workers
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.cache_dir = cache_dir
        self.load_from_cache = load_from_cache
        
    def setup(self, stage: Optional[str] = None):
        dataset_raw=load_dataset('conll2012_ontonotesv5', 'english_v4', cache_dir=self.cache_dir)
        dataset_raw=self.get_doc_parts(dataset_raw) # split each document into document part since the annotation is made with regard of each document part
        self.dataset_train, self.dataset_val, self.dataset_test=self.get_transformed_dataset(dataset_raw)
    
    def get_transformed_dataset(self, dataset_raw: Dataset):
        config={"max_seq_len":4096}
        dataset_train=S2ECorefDataset(dataset_raw["train"], self.tokenizer, config)
        dataset_val=S2ECorefDataset(dataset_raw["validation"], self.tokenizer, config)
        dataset_test=S2ECorefDataset(dataset_raw["test"], self.tokenizer, config)    
        return dataset_train, dataset_val, dataset_test
        
    def get_doc_parts(self, dataset: Dataset):
        return dataset.map(
            self.split_doc_into_docParts,
            batched=True,
            batch_size=1
        )    
   
    @staticmethod
    def split_doc_into_docParts(example):
        """take a doc and return the doc parts"""
        docParts_dict={} #{part0:[], part1:[]...}
        for sent_dict in example["sentences"][0]:
            sent_part_id=sent_dict["part_id"]
            if sent_part_id in docParts_dict:
                docParts_dict[sent_part_id].append(sent_dict)
            else:
                docParts_dict[sent_part_id]=[sent_dict]
        docName=example["document_id"]
        num_parts=len(docParts_dict)
        return {"document_id": [docName]*num_parts, "sentences": [docParts_dict[k] for k in docParts_dict]}

    def train_dataloader(self) -> DataLoader:
        return S2ECorefDataloader(self.dataset_train, num_workers=self.num_workers, config={"max_total_seq_len":5000, "is_training":True})

    def val_dataloader(self) -> DataLoader:
        return S2ECorefDataloader(self.dataset_val, num_workers=self.num_workers, config={"max_total_seq_len":5000, "is_training":False})
    
    def test_dataloader(self) -> DataLoader:
        return S2ECorefDataloader(self.dataset_test, num_workers=self.num_workers, config={"max_total_seq_len":5000, "is_training":False})
        
    def state_dict(self) -> Dict[str, Any]:
        return {'tokenizer': self.tokenizer}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.tokenizer = state_dict['tokenizer']
