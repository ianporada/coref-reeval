import os
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
import random, sys

from pytorch_lightning.utilities.types import EVAL_DATALOADERS
sys.path.append("src")
from utils import flatten

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from pytorch_lightning import _logger as log
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, default_data_collator

class WordLevelCorefDataModule(pl.LightningDataModule):
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
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.cache_dir = cache_dir
        self.load_from_cache = load_from_cache
        
    def setup(self, stage: Optional[str] = None):
        dataset = load_dataset("llm-coref/conll2012_v4_english", cache_dir=self.cache_dir)
        dataset = self.process_data(dataset, stage=stage)
        self.dataset = dataset

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        dataset_split = dataset['train' if stage == 'fit' else 'validation']
        column_names = dataset_split.column_names
        
        # TODO: move to config
        if stage=="fit":
            is_training=True
        else:
            is_training=False
        genres = {"bc": 0, "bn": 1, "mz": 2, "nw": 3, "pt": 4, "tc": 5, "wb": 6}
        config = {"genres": genres, "max_segment_len": 512, "is_training": is_training}
        convert_to_features = partial(self.convert_to_features, tokenizer=self.tokenizer, config=config)
        
         # split each document into document part since the annotation is made with regard of each document part
        dataset=dataset.map(
            self.split_doc_into_docParts,
            batched=True,
            batch_size=1
        )
        
        # map each example to the tensor features, remove existing columns
        return dataset.map(
            convert_to_features,
            num_proc=self.num_workers,
            remove_columns=column_names,
            load_from_cache_file=False #self.load_from_cache #False
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
    
    @staticmethod
    def convert_to_features(example, tokenizer, config):
        """take a doc part and convert to tensors to be input to the model"""
        max_segment_len=config["max_segment_len"]
        genres_dict=config["genres"]
        is_training=config["is_training"]
        
        tokens_list=[] #the list of all words in the doc (num_tokens)
        token_end_list=[] #indicating where words end in the list of subtokens (num_tokens)
        subtokens_list=[] #the list of all subtokens without <cls> and <sep> (num_subtokens)
        sentence_end_list=[] #indicating where sentences end in the list of subtokens (num_subtokens)
        subtoken_map_list=[] #a list of word number in the list of subtokens (num_subtokens)
        speaker_list=[] #list of speakers of all tokens without <cls> and <sep> (num_tokens)
        gold_span_list=[] #list of all coref cluster triplets in the form (cluster_id, word_start_index, word_end_index) (num_gold_spans)
        sent_ids=[] #the sentence id for each token (except <cls> and <sep> tokens) (num_tokens)
        sent_idx=-1
        word_idx=-1
        for sent_dict in example["sentences"]: #get the dictionary for one sentence of doc each time
            speaker=sent_dict["speaker"]
            coref_spans_list=sent_dict["coref_spans"]
            for span in coref_spans_list: #get every cluster triplet of the sentence
                span[1]+=word_idx+1
                span[2]+=word_idx+1
                gold_span_list.append(span)
            for word in sent_dict["words"]: #get a word of the sentence each time
                word_idx+=1
                tokens_list.append(word)
                word_subtokens=tokenizer(word, add_special_tokens=False)["input_ids"]
                token_end_list+=[False]*(len(word_subtokens)-1)+[True]
                speaker_list.append(str(speaker))
                sent_ids.append(sent_idx+1)
                for subtoken in word_subtokens:
                    subtokens_list.append(subtoken)
                    sentence_end_list.append(False)
                    subtoken_map_list.append(word_idx) 
            sent_idx+=1  
            sentence_end_list[-1]=True
        
        #split the doc into segments
        segments_list=[]  #a list of segments of length at most the value of max_segment_length
        segment_subtoken_map_list=[] #a list of word number of each segment  
        curr_idx=0
        prev_token_idx=0
        while curr_idx<len(subtokens_list):
            end_idx=min(curr_idx+max_segment_len-1-2, len(subtokens_list)-1)
            while end_idx>=curr_idx and not sentence_end_list[end_idx]: #split at a sentence end
                end_idx-=1
            if end_idx<curr_idx: #Fail to split at a sentence end (sentence too long), then split at token end
                end_idx=min(curr_idx+max_segment_len-1-2, len(subtokens_list)-1)
                while end_idx>=curr_idx and not token_end_list[end_idx]:
                    end_idx-=1
                if end_idx<curr_idx:
                    print("Fail to split document into segments")
            segment=[tokenizer.cls_token_id]+subtokens_list[curr_idx:end_idx+1]+[tokenizer.sep_token_id]
            segments_list.append(segment)
            subtoken_map=subtoken_map_list[curr_idx: end_idx+1]
            segment_subtoken_map_list.append([-1]+subtoken_map+[-1])
            curr_idx=end_idx+1
            prev_token_idx=subtoken_map[-1]
        
        #padding the segments to get input_ids, input_mask
        input_ids=[]
        input_mask=[]
        for segment in segments_list:
            sent_input_ids=segment.copy()
            sent_input_mask=[1]*len(sent_input_ids)
            while len(sent_input_ids)<max_segment_len:
                sent_input_ids.append(tokenizer.pad_token_id)
                sent_input_mask.append(0)
            input_ids.append(sent_input_ids)
            input_mask.append(sent_input_mask)
        input_ids=np.array(input_ids)
        input_mask=np.array(input_mask)
        
        #get word2subword (num_tokens), a list of tuples indicating which subtokens each token except <cls> and <sep> tokens is mapping to 
        word2subword=[]
        acc_idx=0
        for subtoken_map in segment_subtoken_map_list:
            curr_idx=1
            while curr_idx<len(subtoken_map)-1:
                end_idx=curr_idx
                while subtoken_map[end_idx]==subtoken_map[curr_idx] and end_idx<len(subtoken_map)-1:
                    end_idx+=1
                word2subword.append((curr_idx+acc_idx, end_idx-1+acc_idx))
                curr_idx=end_idx
            acc_idx+=len(subtoken_map)
        assert len(word2subword)==len(tokens_list)
        
        #get the speaker ids
        speaker_ids=[] #(num_tokens)
        speaker_dict={"UNK":0}
        for speaker in speaker_list:
            if speaker not in speaker_dict:
                speaker_dict[speaker]=len(speaker_dict)
        for speaker in speaker_list:
            speaker_ids.append(speaker_dict[speaker])
        
        #get genre ids of the doc
        genre_type=example["document_id"][0][:2]
        genre_id=genres_dict[genre_type]
        
        #get gold labels of the doc
        gold_starts=[] #(num_gold_spans), start token index of every gold span
        gold_ends=[] #(num_gold_spans), end token index of every gold span
        gold_mention_cluster_map=[] #(num_gold_spans)
        cluster_ids=np.array([0]*len(tokens_list)) #cluster id of each token in the doc, non-coreferent token has cluster id 0 (except <cls> and <sep> tokens) (num_tokens)
        for span in gold_span_list:
            gold_mention_cluster_map.append(span[0]+1)
            gold_starts.append(span[1])
            gold_ends.append(span[2])
            cluster_ids[span[1]:span[2]+1]=span[0]+1
        
        #get heads of the gold spans, head of a span is defined as the only word within the span whose head is outside of the span or None. 
        #In case there are no or several such words, the rightmost word of the span is its head
        heads_ids_list=[]
        absolute_id_offset=0
        for sent_dict in example["sentences"]:
            for relative_head_id in sent_dict["head_ids"]:
                if relative_head_id==-1:
                    heads_ids_list.append(-1)
                else:
                    heads_ids_list.append(relative_head_id+absolute_id_offset)
            absolute_id_offset+=len(sent_dict["head_ids"])
        assert len(heads_ids_list)==len(tokens_list)
        
        gold_heads_ids=[] #(num_gold_spans), token index of the head of each gold span
        for (gold_start, gold_end) in zip(gold_starts, gold_ends):
            head_candidates=set()
            for i in range(gold_start, gold_end+1):
                ith_head=heads_ids_list[i]
                if ith_head==-1 or not (gold_start<=ith_head<gold_end):
                    head_candidates.add(i)
            if len(head_candidates)==1:
                gold_heads_ids.append(head_candidates.pop())
            else:
                gold_heads_ids.append(gold_end)
            
        #tensorize all inputs to the model
        input_ids=torch.tensor(input_ids, dtype=torch.long)
        input_mask=torch.tensor(input_mask, dtype=torch.long)
        word2subword=torch.tensor(word2subword, dtype=torch.long)
        speaker_ids=torch.tensor(speaker_ids, dtype=torch.long)
        genre_id=torch.tensor(genre_id, dtype=torch.long)
        sent_ids=torch.tensor(sent_ids, dtype=torch.long)
        gold_starts=torch.tensor(gold_starts, dtype=torch.long)
        gold_ends=torch.tensor(gold_ends, dtype=torch.long)
        gold_mention_cluster_map=torch.tensor(gold_mention_cluster_map, dtype=torch.long)
        cluster_ids=torch.tensor(cluster_ids, dtype=torch.long)
        gold_heads_ids=torch.tensor(gold_heads_ids, dtype=torch.long)
        
        return {"input_ids": input_ids, 
                "input_mask": input_mask, 
                "word2subword": word2subword,
                "speaker_ids": speaker_ids, 
                "genre_id": genre_id, 
                "sent_ids": sent_ids,
                "gold_starts": gold_starts, 
                "gold_ends": gold_ends, 
                "gold_mention_cluster_ids": gold_mention_cluster_map,
                "cluster_ids": cluster_ids,
                "gold_heads_ids": gold_heads_ids,
                "training": is_training}

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset['train'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset['validation'],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset['test'],
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        
    @property
    def collate_fn(self) -> Callable:
        # TODO: move padding to collate function
        return default_data_collator
        
    def state_dict(self) -> Dict[str, Any]:
        return {'tokenizer': self.tokenizer}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.tokenizer = state_dict['tokenizer']
