import sys
sys.path.append('/home/mila/x/xiyuan.zou/research/kd-coref')
import random, sys
from utilities.utils import flatten
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset, load_dataset
from pytorch_lightning import _logger as log
from transformers import PreTrainedTokenizerBase, default_data_collator
from typing import List, Iterable
from nltk.tokenize import sent_tokenize, word_tokenize


def convert_to_features(example, tokenizer, is_training, config): #used for ontonotes
    """take a document and convert to tensors to be input to the model"""
    max_segment_len=config["max_segment_len"]
    max_training_segments=config["max_training_segments"]
    genres_dict=config["genres"]
    tokenizer_type=config["tokenizer_type"]
    
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
    if tokenizer_type=="t5": #T5 tokenizer doesn't add <cls> token, it only adds a <eos> token at the end. So it needs to be treated specially
        while curr_idx<len(subtokens_list):
            end_idx=min(curr_idx+max_segment_len-1-1, len(subtokens_list)-1)
            while end_idx>=curr_idx and not sentence_end_list[end_idx]: #split at a sentence end
                end_idx-=1
            if end_idx<curr_idx: #Fail to split at a sentence end (sentence too long), then split at token end
                end_idx=min(curr_idx+max_segment_len-1-1, len(subtokens_list)-1)
                while end_idx>=curr_idx and not token_end_list[end_idx]:
                    end_idx-=1
                if end_idx<curr_idx:
                    print("Fail to split document into segments")
            segment=subtokens_list[curr_idx:end_idx+1]+[tokenizer.eos_token_id]
            segments_list.append(segment)
            segment_speaker_list.append(speaker_list[curr_idx:end_idx+1]+["SPL"])
            subtoken_map=subtoken_map_list[curr_idx: end_idx+1]
            segment_subtoken_map_list.append(subtoken_map+[-1])
            curr_idx=end_idx+1
            prev_token_idx=subtoken_map[-1]
    else:
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
    if tokenizer_type=="t5":
        for segment in segments_list:
            for i in range(len(segment)-1):
                sentence_map_list.append(sent_idx)
                sent_idx+=int(sentence_end_list[subtok_idx])
                subtok_idx+=1
            sentence_map_list.append(sent_idx) 
    else:
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
    
    return [input_ids, input_mask, speaker_ids, genre_ids, gold_starts, gold_ends, gold_mention_cluster_map, sentence_map, segment_len]


def convert_to_features_GAP(example, tokenizer, config):
    """take a document and convert to tensors to be input to the model"""
    max_segment_len=config["max_segment_len"]
    
    doc_id=example["ID"]
    example["sentences"]=[sent.split() for sent in sent_tokenize(example["Text"])]
    tokens_list=[] #the list of all words in the doc
    token_end_list=[] #indicating where words end in the list of subtokens
    subtokens_list=[] #the list of all subtokens without <cls> and <sep>
    sentence_end_list=[] #indicating where sentences end in the list of subtokens
    subtoken_map_list=[] #a list of word number in the list of subtokens
    word_idx_to_start_token_idx=dict()
    word_idx_to_end_token_idx=dict()
    
    word_idx=-1
    for sent in example["sentences"]: #get one sentence of doc each time
        for word in sent: #get a word of the sentence each time
            word_idx+=1
            word_idx_to_start_token_idx[word_idx]=len(subtokens_list)+1
            tokens_list.append(word)
            word_subtokens=tokenizer(word, add_special_tokens=False)["input_ids"]
            token_end_list+=[False]*(len(word_subtokens)-1)+[True] 
            for subtoken in word_subtokens:
                subtokens_list.append(subtoken)
                sentence_end_list.append(False)
                subtoken_map_list.append(word_idx)
            word_idx_to_end_token_idx[word_idx]=len(subtokens_list)
        sentence_end_list[-1]=True
    
    #split the doc into segments
    segments_list=[]  #a list of segments of length at most the value of max_segment_length
    segment_subtoken_map_list=[] #a list of word number of each segment 
    curr_idx=0
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
    assert len(segments_list)==1 #GAP dataset is short, there should not be more than 1 segment input to c2f model
    
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

    #get segment_len_list
    segment_len_list=np.array([len(s) for s in segments_list]) # a list of effective length for each segment

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
    
    #get GAP gold annotations
    character_offset=0
    for char in example["Text"]:
        word_start_idx=len(example["Text"][:character_offset].split())
        if(character_offset==example["Pronoun-offset"]): #each gold span is in the form (word_start_idx, word_end_idx)
            pronoun_word_idx=(word_start_idx, word_start_idx+len(example["Pronoun"].split())-1)
            pronoun_subtoken_idx=(word_idx_to_start_token_idx[pronoun_word_idx[0]], word_idx_to_end_token_idx[pronoun_word_idx[1]]) #each gold span is in the form (subtoken_start_idx, subtoken_end_idx)
        if(character_offset==example["A-offset"]):
            A_word_idx=(word_start_idx, word_start_idx+len(example["A"].split())-1)
            A_subtoken_idx=(word_idx_to_start_token_idx[A_word_idx[0]], word_idx_to_end_token_idx[A_word_idx[1]])
        if(character_offset==example["B-offset"]):
            B_word_idx=(word_start_idx, word_start_idx+len(example["B"].split())-1)
            B_subtoken_idx=(word_idx_to_start_token_idx[B_word_idx[0]], word_idx_to_end_token_idx[B_word_idx[1]])
        character_offset+=1
        
    assert pronoun_subtoken_idx!=None
    assert A_subtoken_idx!=None
    assert B_subtoken_idx!=None   
    
    #tensorize all inputs to the model
    input_ids=torch.tensor(input_ids, dtype=torch.long)
    input_mask=torch.tensor(input_mask, dtype=torch.long)
    speaker_ids=torch.zeros_like(input_ids)
    segment_len=torch.tensor(segment_len_list, dtype=torch.long)
    genre_ids=torch.tensor(0, dtype=torch.long) 
    sentence_map=torch.tensor(sentence_map_list, dtype=torch.long)
    
    return [input_ids, input_mask, speaker_ids, genre_ids, None, None, None, sentence_map, segment_len, (doc_id, pronoun_subtoken_idx, A_subtoken_idx, B_subtoken_idx)]

def convert_to_features_ontoGUM(example, tokenizer, config):
    """take a document and convert to tensors to be input to the model"""
    max_segment_len=config["max_segment_len"]
    max_training_segments=config["max_training_segments"]
    
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
    
    #genre defaults to 0 for ontoGUM docs
    genre_ids=0
        
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
    
    return [input_ids, input_mask, speaker_ids, genre_ids, gold_starts, gold_ends, gold_mention_cluster_map, sentence_map, segment_len]


def convert_to_features_ontonotes_official(example, tokenizer, config): #used for official evaluation on ontonotes
    """take a document and convert to tensors to be input to the model"""
    max_segment_len=config["max_segment_len"]
    max_training_segments=config["max_training_segments"]
    genres_dict=config["genres"]
    tokenizer_type=config["tokenizer_type"]
    
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
    genre_type=example["document_id"][:2]
    genre_ids=genres_dict[genre_type]
        
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
    
    #info needed for official evaluation
    doc_id=example["document_id"]
    hf_sentences=[]
    global_word_idx_to_local_word_idx={}
    global_word_idx=-1
    for sent_dict in example["sentences"]: #get the dictionary for one sentence of doc each time
        hf_sentences.append({"words": sent_dict["words"]})
        local_word_idx=-1
        for word in sent_dict["words"]:
            global_word_idx+=1
            local_word_idx+=1
            global_word_idx_to_local_word_idx[global_word_idx]=local_word_idx
    
    return [input_ids, input_mask, speaker_ids, genre_ids, gold_starts, gold_ends, gold_mention_cluster_map, sentence_map, segment_len, (doc_id, hf_sentences, sentence_map_list, subtoken_map, global_word_idx_to_local_word_idx)]



class C2F_Dataset(Dataset):
    def __init__(self, dataset_raw: Dataset, tokenizer, is_training, config): #config={"genres", "max_segment_len", "max_training_segments", "tokenizer_type"}
        self.dataset_raw=dataset_raw
        self.tokenizer=tokenizer
        self.is_training=is_training
        self.config=config
        
        self.coref_examples=[] #list of the input of each doc to the model
        for doc in self.dataset_raw:
            doc_input=convert_to_features(doc, self.tokenizer, self.is_training, self.config)
            self.coref_examples.append(doc_input)
            
    def __len__(self):
        return len(self.coref_examples)

    def __getitem__(self, index):
        return self.coref_examples[index]


class C2F_Dataloader(DataLoader):
    def __init__(self, dataset: Dataset, num_workers):
        self.dataset=dataset
        self.num_workers=num_workers
        self.batches=dataset.coref_examples
    
    def __iter__(self) -> Iterable[List[int]]:
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches) 
    

class C2F_Dataset_GAP(Dataset):
    def __init__(self, dataset_raw: Dataset, tokenizer, config): #config={"max_segment_len", "max_training_segments"}
        self.dataset_raw=dataset_raw
        self.tokenizer=tokenizer
        self.config=config
        
        self.coref_examples=[] #list of the input of each doc to the model
        for doc in self.dataset_raw:
            doc_input=convert_to_features_GAP(doc, self.tokenizer, self.config)
            self.coref_examples.append(doc_input)
            
    def __len__(self):
        return len(self.coref_examples)

    def __getitem__(self, index):
        return self.coref_examples[index]
    
    
class C2F_Dataset_ontoGUM(Dataset):
    def __init__(self, dataset_raw: Dataset, tokenizer, config): #config={"max_segment_len", "max_training_segments"}
        self.dataset_raw=dataset_raw
        self.tokenizer=tokenizer
        self.config=config
        
        self.coref_examples=[] #list of the input of each doc to the model
        for doc in self.dataset_raw:
            doc_input=convert_to_features_ontoGUM(doc, self.tokenizer, self.config)
            self.coref_examples.append(doc_input)
            
    def __len__(self):
        return len(self.coref_examples)

    def __getitem__(self, index):
        return self.coref_examples[index]
    
    
class C2F_Dataset_official(Dataset):
    def __init__(self, dataset_raw: Dataset, tokenizer, config): #config={"max_segment_len", "max_training_segments"}
        self.dataset_raw=dataset_raw
        self.tokenizer=tokenizer
        self.config=config
        
        self.coref_examples=[] #list of the input of each doc to the model
        for doc in self.dataset_raw:
            doc_input=convert_to_features_ontonotes_official(doc, self.tokenizer, self.config)
            self.coref_examples.append(doc_input)
            
    def __len__(self):
        return len(self.coref_examples)

    def __getitem__(self, index):
        return self.coref_examples[index]

