import sys
sys.path.append('/home/mila/x/xiyuan.zou/research/kd-coref')
from typing import Any, Callable, Dict, Optional, Union
import random
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from utilities.utils import flatten 
from utilities.consts import SPEAKER_START, SPEAKER_END, NULL_ID_FOR_COREF
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset, load_dataset
from pytorch_lightning import _logger as log
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, default_data_collator
import math
from typing import List, Iterable

class S2E_Dataset(Dataset):
    def __init__(self, dataset_raw: Dataset, tokenizer, tokenizer_type, max_seq_len): 
        self.dataset_raw=dataset_raw
        self.tokenizer=tokenizer
        self.tokenizer_type=tokenizer_type
        self.max_seq_len=max_seq_len
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
            
            if self.tokenizer_type!="t5":
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
            else:
                word_idx_to_start_token_idx=dict()
                word_idx_to_end_token_idx=dict()
                end_token_idx_to_word_idx=[]  
                token_ids_list=[] #token_ids_list(num_subtokens): list of all subtokens in the doc
                last_speaker=None
                self.tokenizer.add_tokens('Ġ#####')
                self.tokenizer.add_tokens('Ġ###')
                SPEAKER_START_ID=self.tokenizer.convert_tokens_to_ids('Ġ#####')
                SPEAKER_END_ID=self.tokenizer.convert_tokens_to_ids('Ġ###')
                for idx, (word, speaker) in enumerate(zip(input_words_list, speakers_list)):
                    if last_speaker!=speaker:
                        speaker_prefix=[SPEAKER_START_ID]+self.tokenizer.encode(" " +speaker, add_special_tokens=False)+[SPEAKER_END_ID]
                        last_speaker=speaker
                    else:
                        speaker_prefix=[]
                    for _ in range(len(speaker_prefix)):
                        end_token_idx_to_word_idx.append(idx)
                    token_ids_list.extend(speaker_prefix)
                    word_idx_to_start_token_idx[idx]=len(token_ids_list)
                    tokenized=self.tokenizer.encode(" " + word, add_special_tokens=False)
                    for _ in range(len(tokenized)):
                        end_token_idx_to_word_idx.append(idx)
                    token_ids_list.extend(tokenized)
                    word_idx_to_end_token_idx[idx]=len(token_ids_list)
            #new_clusters(num_clusters), each span is in the form (subtoken_start_idx, subtoken_end_idx)
            new_clusters=[[(word_idx_to_start_token_idx[start], word_idx_to_end_token_idx[end]) for start, end in cluster] for cluster in clusters_list]
            
            if self.max_seq_len>512 and 0<self.max_seq_len<len(token_ids_list): #if not using leftover batching, filter out all docs longer than the max sequence length allowed
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

    def pad_batch(self, batch, max_length): #used for LLMs taking long context such as longformer
        if self.tokenizer_type!="t5":
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
        else:
            max_length+=1
            input_ids=[]
            input_masks=[]
            gold_clusters=[]
            for example in batch:
                input_ids_per_example=example[0]+[self.tokenizer.eos_token_id]+[self.tokenizer.pad_token_id]*(max_length-len(example[0])-1)
                input_masks_per_example=[1]*(len(example[0])+1)+[0]*(max_length-len(example[0])-1)
                gold_clusters_per_example=self.pad_clusters(example[1])
                input_ids.append(input_ids_per_example)
                input_masks.append(input_masks_per_example)
                gold_clusters.append(gold_clusters_per_example)
            
        input_ids=torch.tensor(input_ids, dtype=torch.long)
        input_masks=torch.tensor(input_masks, dtype=torch.long)
        gold_clusters=torch.tensor(gold_clusters, dtype=torch.long)
        return (input_ids, input_masks, gold_clusters)
    
    def pad_batch_leftover(self, batch, max_length): #used for LLMs taking short context such as roberta
        input_ids, input_masks, gold_clusters=self.pad_batch(batch, max_length) #pad to the longest doc in the batch
        input_ids, input_masks, gold_clusters=input_ids.tolist(), input_masks.tolist(), gold_clusters.tolist()
        
        #break down a doc into segment of segment len
        input_ids=[[doc_ids[i:i+self.max_seq_len] for i in range(0, len(doc_ids), self.max_seq_len)] for doc_ids in input_ids]
        input_masks=[[doc_masks[i:i+self.max_seq_len] for i in range(0, len(doc_masks), self.max_seq_len)] for doc_masks in input_masks]

        #if we have more than 1 segment and the last segment is less than segment_len we have leftovers.
        leftover_input_ids, leftover_input_masks=None, None
        if len(input_ids[0])>1:  #and len(input_ids[0][-1])<self.max_seq_len
            leftover_input_ids=[ids[-1] for ids in input_ids]
            leftover_input_masks=[mask[-1] for mask in input_masks]
            #remove leftovers from main batch
            input_ids=[ids[:-1] for ids in input_ids]
            input_masks=[mask[:-1] for mask in input_masks]
        
        input_ids=torch.tensor(input_ids, dtype=torch.long)
        input_masks=torch.tensor(input_masks, dtype=torch.long)
        gold_clusters=torch.tensor(gold_clusters, dtype=torch.long)
        if leftover_input_ids==None:
            return (input_ids.squeeze(1), input_masks.squeeze(1), gold_clusters)
        else:
            leftover_input_ids=torch.tensor(leftover_input_ids, dtype=torch.long)
            leftover_input_masks=torch.tensor(leftover_input_masks, dtype=torch.long)
            return (input_ids, input_masks, gold_clusters, leftover_input_ids, leftover_input_masks)

    

class S2E_Dataloader(DataLoader):
    def __init__(self, dataset: Dataset, num_workers, max_total_seq_len, is_training): 
        dataset.coref_examples.sort(key=lambda x: len(x[0]), reverse=True)
        self.dataset=dataset
        self.max_total_seq_len=max_total_seq_len
        self.is_training=is_training
        self.num_workers=num_workers
        
        if self.dataset.max_seq_len <= 512: #check if leftover batching is needed
            self.use_leftover=True
        else:
            self.use_leftover=False
            
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
                if self.use_leftover:
                    batch=self.dataset.pad_batch_leftover(batch, len(batch[0][0]))
                else:
                    batch=self.dataset.pad_batch(batch, len(batch[0][0]))
                batches.append(batch)
                batch=[]
                per_example_batch_len=self.calc_effective_per_example_batch_len(len(example[0]))
            batch.append(example)
        if len(batch)==0:
            return batches
        
        if self.use_leftover:
            batch=self.dataset.pad_batch_leftover(batch, len(batch[0][0]))
        else:
            batch=self.dataset.pad_batch(batch, len(batch[0][0]))
        batches.append(batch)
        return batches
    
    def prepare_eval_batches(self):
        batches=[]
        for example in self.dataset.coref_examples:
            if self.use_leftover:
                batch=self.dataset.pad_batch_leftover([example], len(example[0]))
            else:
                batch=self.dataset.pad_batch([example], len(example[0]))
            batches.append(batch)
        return batches
        
    def __iter__(self) -> Iterable[List[int]]:
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
    

class S2E_Dataset_GAP(Dataset):  
    def __init__(self, dataset_raw: Dataset, tokenizer, max_seq_len): 
        #GAP dataset is used for inference only. It doesn't contain speaker info, and no need to pass gold clusters to the model.
        #It only contains very short docs of several sentences. The longest doc of GAP is less than 512 tokens, so no need to perform leftover batching
        self.dataset_raw=dataset_raw
        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len
        self.num_examples_filtered=0
        self.coref_examples=[] #list of the input of each doc to the model
        
        for doc in self.dataset_raw:
            doc_id=doc["ID"]
            input_words_list=doc["Text"].split() #input_words_list(num_words): list of all words in the doc
            token_ids_list=[] #token_ids_list(num_subtokens): list of all subtokens in the doc
            word_idx_to_start_token_idx=dict()
            word_idx_to_end_token_idx=dict()
        
            for idx, word in enumerate(input_words_list):
                word_idx_to_start_token_idx[idx]=len(token_ids_list)+1  # +1 for <s>
                tokenized=self.tokenizer.encode(" "+word, add_special_tokens=False)
                token_ids_list.extend(tokenized)
                word_idx_to_end_token_idx[idx]=len(token_ids_list)

            character_offset=0
            for char in doc["Text"]:
                word_start_idx=len(doc["Text"][:character_offset].split())
                if(character_offset==doc["Pronoun-offset"]): #each gold span is in the form (word_start_idx, word_end_idx)
                    pronoun_word_idx=(word_start_idx, word_start_idx+len(doc["Pronoun"].split())-1)
                    pronoun_subtoken_idx=(word_idx_to_start_token_idx[pronoun_word_idx[0]], word_idx_to_end_token_idx[pronoun_word_idx[1]]) #each gold span is in the form (subtoken_start_idx, subtoken_end_idx)
                if(character_offset==doc["A-offset"]):
                    A_word_idx=(word_start_idx, word_start_idx+len(doc["A"].split())-1)
                    A_subtoken_idx=(word_idx_to_start_token_idx[A_word_idx[0]], word_idx_to_end_token_idx[A_word_idx[1]])
                if(character_offset==doc["B-offset"]):
                    B_word_idx=(word_start_idx, word_start_idx+len(doc["B"].split())-1)
                    B_subtoken_idx=(word_idx_to_start_token_idx[B_word_idx[0]], word_idx_to_end_token_idx[B_word_idx[1]])
                character_offset+=1

            if self.max_seq_len<len(token_ids_list): #GAP's docs are very short, should not be over the max length
                self.num_examples_filtered+=1
                assert False
            
            self.coref_examples.append((token_ids_list, (doc_id, pronoun_subtoken_idx, A_subtoken_idx, B_subtoken_idx)))
            
    def __len__(self):
        return len(self.coref_examples)

    def __getitem__(self, index):
        return self.coref_examples[index]

    def pad_batch(self, batch, max_length):
        max_length+=2  # we have additional two special tokens <s>, </s>
        input_ids=[]
        input_masks=[]
        GAP_gold_pairs_list=[]
        for example in batch:
            input_ids_per_example=[self.tokenizer.cls_token_id]+example[0]+[self.tokenizer.sep_token_id]+[self.tokenizer.pad_token_id]*(max_length-len(example[0])-2)
            input_masks_per_example=[1]*(len(example[0])+2)+[0]*(max_length-len(example[0])-2)
            input_ids.append(input_ids_per_example)
            input_masks.append(input_masks_per_example)
            GAP_gold_pairs_list.append(example[1])
                
        input_ids=torch.tensor(input_ids, dtype=torch.long)
        input_masks=torch.tensor(input_masks, dtype=torch.long)
        return (input_ids, input_masks, None, GAP_gold_pairs_list) #GAP_gold_pairs_list should not be passed to the model 
    
    def pad_batch_leftover(self, batch, max_length): 
        return self.pad_batch(batch, max_length) 
    

class S2E_Dataset_official(Dataset): #for official evaluation on ontonotes
    def __init__(self, dataset_raw: Dataset, tokenizer, max_seq_len): 
        self.dataset_raw=dataset_raw
        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len
        self.max_mention_num=-1
        self.max_cluster_size=-1
        self.max_num_clusters=-1
        self.num_examples_filtered=0
        self.coref_examples=[] #list of the input of each doc to the model
        
        for doc in self.dataset_raw:
            input_words_list=[]
            speakers_list=[]
            sentence_id_list=[]
            clusters_dict={}
            word_acc_idx=0
            sent_id=-1
            for sent_dict in doc["sentences"]:
                sent_id+=1
                input_words_list.append(sent_dict["words"])
                speakers_list.append([str(sent_dict["speaker"])]*len(sent_dict["words"]))
                sentence_id_list.append([sent_id]*len(sent_dict["words"]))
                for span in sent_dict["coref_spans"]:
                    if span[0] in clusters_dict:
                        clusters_dict[span[0]].append((span[1]+word_acc_idx, span[2]+word_acc_idx))
                    else:
                        clusters_dict[span[0]]=[(span[1]+word_acc_idx, span[2]+word_acc_idx)]
                word_acc_idx+=len(sent_dict["words"])
            input_words_list=flatten(input_words_list) #input_words_list(num_words): list of all words in the doc
            speakers_list=flatten(speakers_list) #speakers_list(num_words): list of speakers for each word
            sentence_id_list=flatten(sentence_id_list) #sentence_id_list(num_words): list of sent_id for each word
            clusters_list=[cluster for cluster_id, cluster in clusters_dict.items()] #clusters_list(num_clusters), each span is in the form (word_start_idx, word_end_idx), eg.[[(0,2),(4,5)],[(3,3),(6,9),(15,15)]]
            self.max_mention_num=max(self.max_mention_num, len(flatten(clusters_list)))
            self.max_cluster_size=max(self.max_cluster_size, max(len(cluster) for cluster in clusters_list) if clusters_list else 0)
            self.max_num_clusters=max(self.max_num_clusters, len(clusters_list) if clusters_list else 0)  
            
            word_idx_to_start_token_idx=dict()
            word_idx_to_end_token_idx=dict()
            end_token_idx_to_word_idx=[0]  # for <s>
            token_ids_list=[] #token_ids_list(num_subtokens): list of all subtokens in the doc
            sentence_map_list=[0] #sentence_map_list(num_subtokens): list of sent_id for each subtoken in the doc
            subtoken_map_list=[0] #subtoken_map_list(num_subtokens): list of word_id for each subtoken in the doc
            last_speaker=None
            for idx, (word, speaker, sent_id) in enumerate(zip(input_words_list, speakers_list, sentence_id_list)):
                if last_speaker!=speaker:
                    speaker_prefix=[SPEAKER_START]+self.tokenizer.encode(" " +speaker, add_special_tokens=False)+[SPEAKER_END]
                    last_speaker=speaker
                else:
                    speaker_prefix=[]
                for _ in range(len(speaker_prefix)):
                    end_token_idx_to_word_idx.append(idx)
                    sentence_map_list.append(sent_id)
                    subtoken_map_list.append(idx)
                token_ids_list.extend(speaker_prefix)
                word_idx_to_start_token_idx[idx]=len(token_ids_list)+1  # +1 for <s>
                tokenized=self.tokenizer.encode(" " + word, add_special_tokens=False)
                for _ in range(len(tokenized)):
                    end_token_idx_to_word_idx.append(idx)
                    sentence_map_list.append(sent_id)
                    subtoken_map_list.append(idx)
                token_ids_list.extend(tokenized)
                word_idx_to_end_token_idx[idx]=len(token_ids_list)
            #new_clusters(num_clusters), each span is in the form (subtoken_start_idx, subtoken_end_idx)
            new_clusters=[[(word_idx_to_start_token_idx[start], word_idx_to_end_token_idx[end]) for start, end in cluster] for cluster in clusters_list]
            
            if self.max_seq_len>512 and 0<self.max_seq_len<len(token_ids_list): #if not using leftover batching, filter out all docs longer than the max sequence length allowed
                self.num_examples_filtered+=1
                continue
            
            #info needed for official evaluation
            doc_id=doc["document_id"]
            hf_sentences=[]
            global_word_idx_to_local_word_idx={}
            global_word_idx=-1
            for sent_dict in doc["sentences"]: #get the dictionary for one sentence of doc each time
                hf_sentences.append({"words": sent_dict["words"]})
                local_word_idx=-1
                for word in sent_dict["words"]:
                    global_word_idx+=1
                    local_word_idx+=1
                    global_word_idx_to_local_word_idx[global_word_idx]=local_word_idx
            sentence_map_list.append(sentence_map_list[-1]) #for </s> token
            subtoken_map_list.append(subtoken_map_list[-1])
            self.coref_examples.append((token_ids_list, new_clusters, (doc_id, hf_sentences, sentence_map_list, subtoken_map_list, global_word_idx_to_local_word_idx)))
    
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

    def pad_batch(self, batch, max_length): #used for LLMs taking long context such as longformer
        max_length+=2  # we have additional two special tokens <s>, </s>
        input_ids=[]
        input_masks=[]
        gold_clusters=[]
        official_eval_info=[]
        for example in batch:
            input_ids_per_example=[self.tokenizer.cls_token_id]+example[0]+[self.tokenizer.sep_token_id]+[self.tokenizer.pad_token_id]*(max_length-len(example[0])-2)
            input_masks_per_example=[1]*(len(example[0])+2)+[0]*(max_length-len(example[0])-2)
            gold_clusters_per_example=self.pad_clusters(example[1])
            input_ids.append(input_ids_per_example)
            input_masks.append(input_masks_per_example)
            gold_clusters.append(gold_clusters_per_example)
            official_eval_info.append(example[2])
            
        input_ids=torch.tensor(input_ids, dtype=torch.long)
        input_masks=torch.tensor(input_masks, dtype=torch.long)
        gold_clusters=torch.tensor(gold_clusters, dtype=torch.long)
        return (input_ids, input_masks, gold_clusters, official_eval_info)
    
    def pad_batch_leftover(self, batch, max_length): #used for LLMs taking short context such as roberta
        input_ids, input_masks, gold_clusters, official_eval_info=self.pad_batch(batch, max_length) #pad to the longest doc in the batch
        input_ids, input_masks, gold_clusters=input_ids.tolist(), input_masks.tolist(), gold_clusters.tolist()
        
        #break down a doc into segment of segment len
        input_ids=[[doc_ids[i:i+self.max_seq_len] for i in range(0, len(doc_ids), self.max_seq_len)] for doc_ids in input_ids]
        input_masks=[[doc_masks[i:i+self.max_seq_len] for i in range(0, len(doc_masks), self.max_seq_len)] for doc_masks in input_masks]

        #if we have more than 1 segment and the last segment is less than segment_len we have leftovers.
        leftover_input_ids, leftover_input_masks=None, None
        if len(input_ids[0])>1:  #and len(input_ids[0][-1])<self.max_seq_len
            leftover_input_ids=[ids[-1] for ids in input_ids]
            leftover_input_masks=[mask[-1] for mask in input_masks]
            #remove leftovers from main batch
            input_ids=[ids[:-1] for ids in input_ids]
            input_masks=[mask[:-1] for mask in input_masks]
        
        input_ids=torch.tensor(input_ids, dtype=torch.long)
        input_masks=torch.tensor(input_masks, dtype=torch.long)
        gold_clusters=torch.tensor(gold_clusters, dtype=torch.long)
        if leftover_input_ids==None:
            return (input_ids.squeeze(1), input_masks.squeeze(1), gold_clusters, official_eval_info)
        else:
            leftover_input_ids=torch.tensor(leftover_input_ids, dtype=torch.long)
            leftover_input_masks=torch.tensor(leftover_input_masks, dtype=torch.long)
            return (input_ids, input_masks, gold_clusters, leftover_input_ids, leftover_input_masks, official_eval_info)