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


class BertCorefDataModule(pl.LightningDataModule):
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
        dataset = load_dataset('conll2012_ontonotesv5', 'english_v4', cache_dir=self.cache_dir)
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
        config = {"genres": genres, "max_segment_len": 128, "max_training_segments": 11, "is_training": is_training}
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
            load_from_cache_file=self.load_from_cache #False
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
        max_training_segments=config["max_training_segments"]
        genres_dict=config["genres"]
        is_training=config["is_training"]
        
        tokens_list=[] #the list of all words in the doc
        token_end_list=[] #indicating where words end in the list of subtokens
        subtokens_list=[] #the list of all subtokens without <cls> and <sep>
        sentence_end_list=[] #indicating where sentences end in the list of subtokens
        subtoken_map_list=[] #a list of word number in the list of subtokens
        speaker_list=[] #list of speaker of all subtokens without <cls> and <sep>
        gold_span_list=[] #list of all coref cluster triplets in the form (cluster_id, word_start_index, word_end_index)
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
                for subtoken in word_subtokens:
                    subtokens_list.append(subtoken)
                    sentence_end_list.append(False)
                    subtoken_map_list.append(word_idx)
                    speaker_list.append(str(speaker))
            sentence_end_list[-1]=True
        
        #split the doc into segments
        segments_list=[]  #a list of segments of length at most the value of max_segment_length
        segment_subtoken_map_list=[] #a list of word number of each segment
        segment_speaker_list=[] #stores the speaker of every subtoken of every segment, <SPL> for <cls> and <sep> token  
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
            segment_speaker_list.append(["SPL"]+speaker_list[curr_idx:end_idx+1]+["SPL"])
            subtoken_map=subtoken_map_list[curr_idx: end_idx+1]
            segment_subtoken_map_list.append([-1]+subtoken_map+[-1])
            curr_idx=end_idx+1
            prev_token_idx=subtoken_map[-1]
    
        #get the gold span
        gold_starts=[]
        gold_ends=[]
        gold_mention_cluster_map=[] 
        subtoken_map=flatten(segment_subtoken_map_list)
        for span in gold_span_list:
            word_start_idx=span[1]
            word_end_idx=span[2]
            subtoken_start_idx=subtoken_map.index(word_start_idx)
            subtoken_end_idx=len(subtoken_map)-subtoken_map[::-1].index(word_end_idx)-1
            gold_starts.append(subtoken_start_idx)
            gold_ends.append(subtoken_end_idx)
            gold_mention_cluster_map.append(span[0]+1)
        gold_starts=np.array(gold_starts)
        gold_ends=np.array(gold_ends)
        gold_mention_cluster_map=np.array(gold_mention_cluster_map)
        
        #get the sentence map
        sentence_map_list=[] #indicating the sentence number of every subtokens including <cls> and <sep>
        sent_idx, subtok_idx=0, 0
        for segment in segments_list:
            sentence_map_list.append(sent_idx)  
            for i in range(len(segment)-2):
                sentence_map_list.append(sent_idx)
                sent_idx+=int(sentence_end_list[subtok_idx])
                subtok_idx+=1
            sentence_map_list.append(sent_idx) 
       
        #get the speaker dictionary
        speaker_dict={"UNK":0, "SPL":1}
        for speaker in speaker_list:
            if speaker not in speaker_dict:
                speaker_dict[speaker]=len(speaker_dict)
    
        #get segment_len_list
        total_num_subtokens=sum([len(s) for s in segments_list]) #total number of effective subtokens of the doc
        segment_len_list=np.array([len(s) for s in segments_list]) # a list of effective length for each segment
    
        #padding the segments to get input_ids, input_mask, speaker_ids
        input_ids=[]
        input_mask=[]
        speaker_ids=[]
        for (segment, speakers) in zip(segments_list, segment_speaker_list):
            sent_input_ids=segment.copy()
            sent_input_mask=[1]*len(sent_input_ids)
            sent_speaker_ids=[speaker_dict[speaker] for speaker in speakers]
            while len(sent_input_ids)<max_segment_len:
                sent_input_ids.append(tokenizer.pad_token_id)
                sent_input_mask.append(0)
                sent_speaker_ids.append(0)
            input_ids.append(sent_input_ids)
            input_mask.append(sent_input_mask)
            speaker_ids.append(sent_speaker_ids)
        input_ids=np.array(input_ids)
        input_mask=np.array(input_mask)
        speaker_ids=np.array(speaker_ids)
        
        #get genre ids of the doc
        genre_type=example["document_id"][0][:2]
        genre_ids=genres_dict[genre_type]

        #truncate doc into max_training_segments number of continuous segments from the doc
        if is_training and len(segments_list)>max_training_segments:
            num_segments=input_ids.shape[0]
            sent_offset=random.randint(0, num_segments-max_training_segments)
            word_offset=segment_len_list[:sent_offset].sum()
            num_words=segment_len_list[sent_offset:sent_offset+max_training_segments].sum()
            
            input_ids=input_ids[sent_offset:sent_offset+max_training_segments, :]
            input_mask=input_mask[sent_offset:sent_offset+max_training_segments, :]
            speaker_ids=speaker_ids[sent_offset:sent_offset+max_training_segments, :]
            segment_len_list=segment_len_list[sent_offset:sent_offset+max_training_segments]
            sentence_map_list=sentence_map_list[word_offset:word_offset+num_words]
            gold_spans=(gold_starts<word_offset+num_words) & (gold_ends>=word_offset)
            gold_starts=gold_starts[gold_spans]-word_offset
            gold_ends=gold_ends[gold_spans]-word_offset
            gold_mention_cluster_map=gold_mention_cluster_map[gold_spans]
            
        #tensorize all inputs to the model
        input_ids=torch.tensor(input_ids, dtype=torch.long)
        input_mask=torch.tensor(input_mask, dtype=torch.long)
        speaker_ids=torch.tensor(speaker_ids, dtype=torch.long)
        segment_len=torch.tensor(segment_len_list, dtype=torch.long)
        genre_ids=torch.tensor(genre_ids, dtype=torch.long)
        sentence_map=torch.tensor(sentence_map_list, dtype=torch.long)
        gold_starts=torch.tensor(gold_starts, dtype=torch.long)
        gold_ends=torch.tensor(gold_ends, dtype=torch.long)
        gold_mention_cluster_map=torch.tensor(gold_mention_cluster_map, dtype=torch.long)
        
        return {"input_ids": input_ids, 
                "input_mask": input_mask, 
                "speaker_ids": speaker_ids, 
                "genre_ids": genre_ids, 
                "gold_starts": gold_starts, 
                "gold_ends": gold_ends, 
                "gold_mention_cluster_ids": gold_mention_cluster_map,
                "sentence_map": sentence_map,
                "segment_len": segment_len}

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
