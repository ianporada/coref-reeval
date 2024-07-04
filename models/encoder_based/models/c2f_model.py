import sys
sys.path.append('/home/mila/x/xiyuan.zou/research/kd-coref')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import AutoModel, T5EncoderModel
from collections import Iterable
from typing import Any, Dict, Optional, Type
from utilities.utils import bucket_distance, batch_select
    
""" config: {"feature_emb_size":int, "dropout_rate":float, "llm_pretrained_name":str, "llm_type":str, "max_span_width":int, 
    "max_segment_len":int, "use_span_width_to_compute_emb":bool, "span_emb_compute_method":str, "ffnn_size":int, "ffnn_depth":int,
    "max_num_extracted_spans":int, "top_span_ratio":float, "crossing_mentions_allowed":bool, "max_top_antecedents":int,
    "use_span_dist_to_compute_rough_score":bool, "use_fine_score":bool, "use_speaker_info_to_compute_fine_score":bool,
    "use_genre_info_to_compute_fine_score":bool, "use_seg_dist_to_compute_fine_score":bool, "use_antecedent_dist_to_compute_fine_score":bool,
    "num_genres":int, "max_num_segments":int}
"""

class C2F_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config        
        self.max_span_width=config["max_span_width"]
        self.max_seg_len=config["max_segment_len"]
        
        self.dropout=nn.Dropout(p=self.config["dropout_rate"])
        if config["llm_type"]!='t5':
            self.llm_encoder=AutoModel.from_pretrained(self.config["llm_pretrained_name"])
        else:
            self.llm_encoder=T5EncoderModel.from_pretrained(self.config["llm_pretrained_name"])
        self.encoder_emb_size=self.llm_encoder.config.hidden_size
        self.span_emb_size=self.encoder_emb_size*3
        if(self.config["use_span_width_to_compute_emb"]):
            self.span_emb_size+=self.config["feature_emb_size"]
        self.pair_emb_size=self.span_emb_size*3
        if(self.config["use_speaker_info_to_compute_fine_score"]):
            self.pair_emb_size+=self.config["feature_emb_size"]
        if(self.config["use_genre_info_to_compute_fine_score"]):
            self.pair_emb_size+=self.config["feature_emb_size"]
        if(self.config["use_seg_dist_to_compute_fine_score"]):
            self.pair_emb_size+=self.config["feature_emb_size"]
        if(self.config["use_antecedent_dist_to_compute_fine_score"]):
            self.pair_emb_size+=self.config["feature_emb_size"]
            
        self.emb_span_width=self.make_embedding(self.max_span_width) if self.config["use_span_width_to_compute_emb"] else None
        self.span_emb_attn=self.make_ffnn(self.encoder_emb_size, 0, 1) if self.config["span_emb_compute_method"]=="attention" else None
        self.mention_score_ffnn=self.make_ffnn(self.span_emb_size, [self.config["ffnn_size"]]*self.config["ffnn_depth"],1)
        self.coarse_bilinear=self.make_ffnn(self.span_emb_size,0,self.span_emb_size)
        self.emb_antecedent_distance_prior=self.make_embedding(10) if self.config["use_span_dist_to_compute_rough_score"] else None
        self.antecedent_distance_score_ffnn=self.make_ffnn(self.config["feature_emb_size"], 0, 1) if self.config["use_span_dist_to_compute_rough_score"] else None
        self.emb_same_speaker=self.make_embedding(2) if self.config["use_speaker_info_to_compute_fine_score"] else None
        self.emb_genre=self.make_embedding(self.config["num_genres"]) if self.config["use_genre_info_to_compute_fine_score"] else None
        self.emb_segment_distance=self.make_embedding(self.config["max_num_segments"]) if self.config["use_seg_dist_to_compute_fine_score"] else None
        self.emb_top_antecedent_distance=self.make_embedding(10) if self.config["use_antecedent_dist_to_compute_fine_score"] else None
        self.fine_score_ffnn=self.make_ffnn(self.pair_emb_size, [self.config["ffnn_size"]]*self.config["ffnn_depth"], 1) if self.config["use_fine_score"] else None
        self.emb_span_width_prior=self.make_embedding(self.max_span_width) if self.config["use_span_width_to_compute_mention_score"] else None
        self.span_width_score_ffnn=self.make_ffnn(self.config['feature_emb_size'], [self.config['ffnn_size']]*self.config['ffnn_depth'], 1) if self.config["use_span_dist_to_compute_rough_score"] else None
      
    def make_embedding(self, dict_size, std=0.02): #return an embedding layer
        emb=nn.Embedding(dict_size, self.config["feature_emb_size"])
        init.normal_(emb.weight, std=std)
        return emb
    
    def make_linear(self, in_features, out_features, bias=True, std=0.02): #return a linear layer
        linear=nn.Linear(in_features, out_features, bias)
        init.normal_(linear.weight, std=std)
        if(bias):
            init.zeros_(linear.bias)
        return linear
    
    def make_ffnn(self, feat_size, hidden_size, output_size): #return a MLP
        if(hidden_size is None or hidden_size==0 or hidden_size==[] or hidden_size==[0]):
            return self.make_linear(feat_size, output_size)
        if(not isinstance(hidden_size, Iterable)):
            hidden_size=[hidden_size]
        ffnn=[self.make_linear(feat_size,hidden_size[0]),nn.ReLU(),self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn+=[self.make_linear(hidden_size[i-1],hidden_size[i]),nn.ReLU(),self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1],output_size))
        return nn.Sequential(*ffnn)
    
    def extract_top_non_crossing_spans_by_mention_scores(self, candidate_ids_sorted_by_score, candidate_starts, candidate_ends, num_top_spans):
        #Keep top non-crossing spans ordered by mention scores, compute on CPU because of loop
        selected_candidate_ids=[]
        start_to_max_end, end_to_min_start={},{} #start_to_max_end[i] indicates the max end of the previous spans starting from index i, end_to_min_start[i] indicates the min start of the previous spans ending at index i
        for candidate_ids in candidate_ids_sorted_by_score:
            if(len(selected_candidate_ids)>=num_top_spans):
                break
            # Perform overlapping check
            span_start_ids=candidate_starts[candidate_ids]
            span_end_ids=candidate_ends[candidate_ids]
            cross_overlap=False
            for token_ids in range(span_start_ids, span_end_ids+1):
                max_end=start_to_max_end.get(token_ids, -1)
                if(token_ids>span_start_ids and max_end>span_end_ids):
                    cross_overlap=True
                    break
                min_start=end_to_min_start.get(token_ids, -1)
                if(token_ids<span_end_ids and 0<=min_start<span_start_ids):
                    cross_overlap=True
                    break
            if(not cross_overlap):
                # Pass check; select index and update dict stats
                selected_candidate_ids.append(candidate_ids)
                max_end=start_to_max_end.get(span_start_ids, -1)
                if(span_end_ids>max_end):
                    start_to_max_end[span_start_ids]=span_end_ids
                min_start=end_to_min_start.get(span_end_ids, -1)
                if(min_start==-1 or span_start_ids<min_start):
                    end_to_min_start[span_end_ids]=span_start_ids
        #Sort selected candidates by span ids
        selected_candidate_ids=sorted(selected_candidate_ids, key=lambda ids: (candidate_starts[ids], candidate_ends[ids]))
        if len(selected_candidate_ids) < num_top_spans:  # Padding
            selected_candidate_ids+=([selected_candidate_ids[0]]*(num_top_spans-len(selected_candidate_ids)))
        return selected_candidate_ids
    
    def get_input_emb(self, input_ids, input_mask, speaker_ids):
        output_dict=self.llm_encoder(input_ids, attention_mask=input_mask, return_dict=True) #token_emb(num_segments,sentence_max_len,emb_size)
        token_emb=output_dict["last_hidden_state"]
        input_mask=input_mask.to(torch.bool)
        token_emb=token_emb[input_mask] #token_emb(num_of_tokens,emb_size)
        speaker_ids=speaker_ids[input_mask] #speaker_ids(num_of_tokens)
        num_tokens=token_emb.shape[0]
        return token_emb, speaker_ids, num_tokens, input_mask
        
    def get_all_possible_spans(self, num_tokens, sentence_map):
        candidate_starts=torch.unsqueeze(torch.arange(0,num_tokens,device=self.device),1).repeat(1,self.max_span_width) #candidate_starts(num_tokens,max_span_width)
        candidate_ends=candidate_starts+torch.arange(0,self.max_span_width,device=self.device)
        candidate_starts_ids=sentence_map[candidate_starts] #segment each start and end token belongs to
        candidate_ends_ids=sentence_map[torch.min(candidate_ends,torch.tensor(num_tokens-1,device=self.device))]
        candidate_mask=(candidate_ends<num_tokens) & (candidate_starts_ids==candidate_ends_ids) #keep spans within the same sentence
        candidate_starts, candidate_ends=candidate_starts[candidate_mask], candidate_ends[candidate_mask] #candidate_starts,candidate_ends(num_candidate_spans)
        num_candidate_spans=candidate_starts.shape[0]
        return candidate_starts, candidate_ends, num_candidate_spans
    
    def get_candidate_labels(self, gold_starts, gold_ends, candidate_starts, candidate_ends, gold_mention_cluster_ids):
        same_start=(torch.unsqueeze(gold_starts, 1)==torch.unsqueeze(candidate_starts, 0)) #same_start,same_end(num_gold_spans, num_candidate_spans), whether each gold span and each possible span have the same start and end
        same_end=(torch.unsqueeze(gold_ends, 1)==torch.unsqueeze(candidate_ends, 0))
        same_span=(same_start & same_end).to(torch.long)
        candidate_labels=torch.matmul(torch.unsqueeze(gold_mention_cluster_ids, 0).to(torch.float), same_span.to(torch.float))
        candidate_labels=torch.squeeze(candidate_labels.to(torch.long), 0) #candidate_labels(num_candidate_spans), gold span has cluster id as label, non-gold span has label 0
        return candidate_labels

    def get_span_emb(self, token_emb, candidate_starts, candidate_ends, num_tokens, num_candidate_spans):
        span_start_emb, span_end_emb=token_emb[candidate_starts], token_emb[candidate_ends] #span_start_emb,span_end_emb(num_candidate_spans,emb_size)
        candidate_emb_list=[span_start_emb,span_end_emb]
        if(self.config["use_span_width_to_compute_emb"]): 
            candidate_width_idx=candidate_ends-candidate_starts 
            candidate_width_emb=self.emb_span_width(candidate_width_idx)
            candidate_width_emb=self.dropout(candidate_width_emb)
            candidate_emb_list.append(candidate_width_emb)
        candidate_tokens=torch.unsqueeze(torch.arange(0, num_tokens, device=self.device), 0).repeat(num_candidate_spans, 1)
        candidate_tokens_mask=(candidate_tokens>=torch.unsqueeze(candidate_starts, 1)) & (candidate_tokens<=torch.unsqueeze(candidate_ends, 1)) #candidate_tokens_mask(num_candidate_spans,num_tokens), mask all tokens not in the span for every span
        if(self.config["span_emb_compute_method"]=="attention"): #token_attn(num_tokens), preliminary weights of every token
            token_attn=torch.squeeze(self.span_emb_attn(token_emb), 1)
        elif(self.config["span_emb_compute_method"]=="average"):
            token_attn=torch.ones(num_tokens, dtype=torch.float, device=self.device)
        candidate_tokens_attn_raw=torch.log(candidate_tokens_mask.to(torch.float))+torch.unsqueeze(token_attn, 0) #for every candidate span. torch.log turns the weights of masked tokens to nearly zero 
        candidate_tokens_attn=nn.functional.softmax(candidate_tokens_attn_raw, dim=1) #candidate_tokens_attn(num_candidate_spans,num_tokens), assigns attention weights for tokens in every candidate span
        head_attn_emb=torch.matmul(candidate_tokens_attn, token_emb) #head_attn_emb(num_candidate_spans,emb_size)
        candidate_emb_list.append(head_attn_emb)
        candidate_span_emb=torch.cat(candidate_emb_list, dim=1) 
        return candidate_span_emb, candidate_width_idx
    
    def get_mention_score(self, candidate_span_emb, candidate_width_idx):
        candidate_mention_scores=torch.squeeze(self.mention_score_ffnn(candidate_span_emb),1) #candidate_mention_scores(num_candidate_spans,1)
        if self.config['use_span_width_to_compute_mention_score']:
            width_score=torch.squeeze(self.span_width_score_ffnn(self.emb_span_width_prior.weight), 1)
            candidate_width_score=width_score[candidate_width_idx]
            candidate_mention_scores+=candidate_width_score
        return candidate_mention_scores
    
    def get_top_spans_with_highest_mention_scores(self, candidate_mention_scores, candidate_starts, candidate_ends, num_tokens, candidate_span_emb, candidate_labels, do_loss):
        candidate_ids_sorted_by_score=torch.argsort(candidate_mention_scores, descending=True).tolist()
        candidate_starts_cpu, candidate_ends_cpu=candidate_starts.tolist(), candidate_ends.tolist()
        num_top_spans=int(min(self.config["max_num_extracted_spans"], self.config["top_span_ratio"]*num_tokens))
        num_top_spans=max(num_top_spans,2) #make sure the number of top spans always >=2
        if(self.config["crossing_mentions_allowed"]):
            selected_ids_cpu=candidate_ids_sorted_by_score[:num_top_spans]
        else:
            selected_ids_cpu=self.extract_top_non_crossing_spans_by_mention_scores(candidate_ids_sorted_by_score,candidate_starts_cpu,candidate_ends_cpu,num_top_spans)
        selected_ids=torch.tensor(selected_ids_cpu, device=self.device)
        top_span_starts, top_span_ends=candidate_starts[selected_ids], candidate_ends[selected_ids] #top_span_starts,top_span_ends(num_top_spans)
        top_span_emb=candidate_span_emb[selected_ids] #top_span_emb(num_top_spans,span_emb_size)
        top_span_mention_scores=candidate_mention_scores[selected_ids] #top_span_mention_scores(num_top_spans)
        top_span_cluster_ids=candidate_labels[selected_ids] if do_loss else None #top_span_cluster_ids(num_top_spans)
        return top_span_starts, top_span_ends, top_span_emb, top_span_mention_scores, top_span_cluster_ids, num_top_spans

    def get_coarse_score(self, top_span_mention_scores, top_span_emb, num_top_spans, device):
        max_top_antecedent=min(num_top_spans, self.config["max_top_antecedents"])
        top_span_range=torch.arange(0,num_top_spans,device=device)
        antecedent_offsets=torch.unsqueeze(top_span_range,1)-torch.unsqueeze(top_span_range,0)
        antecedent_mask=antecedent_offsets>=1 #antecedent_mask(num_top_spans,num_top_spans), mask all invalid antecedents for each span
        pairwise_mention_score_sum=torch.unsqueeze(top_span_mention_scores,1)+torch.unsqueeze(top_span_mention_scores,0)
        source_span_emb=self.dropout(self.coarse_bilinear(top_span_emb)) #source_span_emb(num_top_spans,span_emb_size)
        target_span_emb=self.dropout(torch.transpose(top_span_emb,0,1)) #target_span_emb(span_emb_size,num_top_spans)
        pairwise_coref_score=torch.matmul(source_span_emb,target_span_emb) #pairwise_coref_score(num_top_spans,num_top_spans)
        pairwise_rough_score=pairwise_coref_score+pairwise_mention_score_sum+torch.log(antecedent_mask.to(torch.float)) #pairwise_rough_score(num_top_spans,num_top_spans)
        if(self.config["use_span_dist_to_compute_rough_score"]):
            distance_score=torch.squeeze(self.antecedent_distance_score_ffnn(self.dropout(self.emb_antecedent_distance_prior.weight)), 1)
            bucketed_distance=bucket_distance(antecedent_offsets)
            antecedent_distance_score=distance_score[bucketed_distance]
            pairwise_rough_score+=antecedent_distance_score
        top_pairwise_rough_score, top_antecedent_ids=torch.topk(pairwise_rough_score,k=max_top_antecedent) #top_pairwise_rough_score,top_antecedent_ids(num_top_spans,max_top_antecedent)
        top_antecedent_mask=batch_select(antecedent_mask, top_antecedent_ids, device)  #top_antecedent_mask,top_antecedent_offsets(num_top_spans, max_top_antecedents)
        top_antecedent_offsets=batch_select(antecedent_offsets, top_antecedent_ids, device)
        return top_pairwise_rough_score, top_antecedent_ids, top_antecedent_mask, top_antecedent_offsets, max_top_antecedent

    def get_final_score(self, input_ids, input_mask, speaker_ids, genre_ids, top_span_starts, top_span_emb, top_antecedent_ids, top_antecedent_offsets, top_pairwise_rough_score, num_top_spans, max_top_antecedent, device):
        if(self.config["use_fine_score"]):
            same_speaker_emb, genre_emb, seg_distance_emb, top_antecedent_distance_emb=None, None, None, None
            if(self.config["use_speaker_info_to_compute_fine_score"]):
                top_span_speaker_ids=speaker_ids[top_span_starts]
                top_antecedent_speaker_ids=top_span_speaker_ids[top_antecedent_ids]
                same_speaker=torch.unsqueeze(top_span_speaker_ids, 1)==top_antecedent_speaker_ids
                same_speaker_emb=self.emb_same_speaker(same_speaker.to(torch.long)) #same_speaker_emb(num_top_spans, max_top_antecedents,emb_size)
            if(self.config["use_genre_info_to_compute_fine_score"]):
                genre_emb=self.emb_genre(genre_ids)
                genre_emb=torch.unsqueeze(torch.unsqueeze(genre_emb, 0), 0).repeat(num_top_spans, max_top_antecedent, 1) #genre_emb(num_top_spans,max_top_antecedents,emb_size)
            if(self.config["use_seg_dist_to_compute_fine_score"]):
                num_segs, seg_len=input_ids.shape[0], input_ids.shape[1]
                token_seg_ids=torch.arange(0, num_segs, device=device).unsqueeze(1).repeat(1, seg_len)
                token_seg_ids=token_seg_ids[input_mask]
                top_span_seg_ids=token_seg_ids[top_span_starts]
                top_antecedent_seg_ids=token_seg_ids[top_span_starts[top_antecedent_ids]]
                top_antecedent_seg_distance=torch.unsqueeze(top_span_seg_ids, 1)-top_antecedent_seg_ids
                top_antecedent_seg_distance=torch.clamp(top_antecedent_seg_distance, 0, self.config["max_num_segments"]-1)
                seg_distance_emb=self.emb_segment_distance(top_antecedent_seg_distance) #seg_distance_emb(num_top_spans,max_top_antecedents,emb_size)
            if(self.config["use_antecedent_dist_to_compute_fine_score"]):
                top_antecedent_distance=bucket_distance(top_antecedent_offsets)
                top_antecedent_distance_emb=self.emb_top_antecedent_distance(top_antecedent_distance) #top_antecedent_distance_emb(num_top_spans,max_top_antecedents,emb_size)
            feature_emb_list=[same_speaker_emb,genre_emb,seg_distance_emb,top_antecedent_distance_emb]
            feature_emb=self.dropout(torch.cat(feature_emb_list,dim=-1))
            top_antecedent_emb=top_span_emb[top_antecedent_ids] #top_antecedent_emb(num_top_spans,max_top_antecedents,span_emb_size)
            source_emb=torch.unsqueeze(top_span_emb,1).repeat(1,max_top_antecedent,1) #source_emb(num_top_spans,max_top_antecedents,span_emb_size)
            similarity_emb=top_antecedent_emb*source_emb #similarity_emb(num_top_spans,max_top_antecedents,span_emb_size)
            pair_emb=torch.cat([source_emb, top_antecedent_emb, similarity_emb, feature_emb],dim=-1)
            top_pairwise_fine_score=torch.squeeze(self.fine_score_ffnn(pair_emb), -1) #top_pairwise_fine_score(num_top_spans,max_top_antecedents)
            top_pairwise_score=top_pairwise_rough_score+top_pairwise_fine_score #top_pairwise_score(num_top_spans,max_top_antecedents)
        else:
            top_pairwise_score=top_pairwise_rough_score
        return top_pairwise_score

    def get_gold_labels(self, top_span_cluster_ids, top_antecedent_ids, top_antecedent_mask):
        top_antecedent_cluster_ids=top_span_cluster_ids[top_antecedent_ids]
        top_antecedent_cluster_ids+=(top_antecedent_mask.to(torch.long)-1)*100000  #Mask id on invalid antecedents

        same_gold_cluster_indicator=(top_antecedent_cluster_ids==torch.unsqueeze(top_span_cluster_ids, 1))
        non_dummy_indicator=torch.unsqueeze(top_span_cluster_ids>0, 1)
        pairwise_labels=same_gold_cluster_indicator & non_dummy_indicator
        dummy_antecedent_labels=torch.logical_not(pairwise_labels.any(dim=1, keepdims=True))
        top_antecedent_gold_labels=torch.cat([dummy_antecedent_labels, pairwise_labels], dim=1) #top_antecedent_gold_labels(num_top_spans, max_top_antecedents+1), label of gold span pair is 1
        return top_antecedent_gold_labels
    
    def get_loss(self, top_pairwise_score, top_antecedent_gold_labels, num_top_spans, device):
        top_antecedent_scores=torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_score], dim=1)
        log_marginalized_antecedent_scores=torch.logsumexp(top_antecedent_scores+torch.log(top_antecedent_gold_labels.to(torch.float)), dim=1)
        log_norm=torch.logsumexp(top_antecedent_scores, dim=1) #regularization term
        loss=torch.sum(-log_marginalized_antecedent_scores+log_norm) #equivalent to maximize the probablity of every gold antecedent for each gold mention 
        return loss, top_antecedent_scores, log_marginalized_antecedent_scores, log_norm

    def forward(self, input_ids, input_mask, speaker_ids, genre_ids, gold_starts, gold_ends, gold_mention_cluster_ids, sentence_map, segment_len): 
        """input_ids(num_segments,sentence_max_len), input_mask(num_segments,sentence_max_len), speaker_ids(num_segments,sentence_max_len)
           genre_ids: int, gold_starts(num_gold_spans), gold_ends(num_gold_spans), gold_mention_cluster_map(num_gold_spans)
           input_ids: token ids of every input sentence, input_mask: token masks of every input sentence
           speaker_ids: speaker_id of every token of every input sentence, genre_ids: the genre of input document
           gold_starts: start of every gold span, gold_ends: end of every gold span, gold_mention_cluster_ids: cluster id of every gold span
        """
        self.device=input_ids.device
        device=self.device

        #Check whether to compute loss
        do_loss=False
        if(gold_mention_cluster_ids is not None):
            assert gold_starts is not None
            assert gold_ends is not None
            do_loss=True
        
        #get the token embedding of input
        token_emb, speaker_ids, num_tokens, input_mask=self.get_input_emb(input_ids, input_mask, speaker_ids)

        #get all possible spans in the input segments
        candidate_starts, candidate_ends, num_candidate_spans=self.get_all_possible_spans(num_tokens, sentence_map)
        
        #get candidate labels
        if do_loss:
            candidate_labels=self.get_candidate_labels(gold_starts, gold_ends, candidate_starts, candidate_ends, gold_mention_cluster_ids)
        else:
            candidate_labels=None
        
        #get span embedding for all candidate spans. Every span embedding is [start token emb,end token emb,span width emb,span attn emb]
        candidate_span_emb, candidate_width_idx=self.get_span_emb(token_emb, candidate_starts, candidate_ends, num_tokens, num_candidate_spans)
        
        #get mention score
        candidate_mention_scores=self.get_mention_score(candidate_span_emb, candidate_width_idx)
        
        #Keep top spans with highest mention scores
        top_span_starts, top_span_ends, top_span_emb, top_span_mention_scores, top_span_cluster_ids, num_top_spans=self.get_top_spans_with_highest_mention_scores(candidate_mention_scores, candidate_starts, candidate_ends, num_tokens, candidate_span_emb, candidate_labels, do_loss)
            
        #get coarse corefered score for each antecedent of each mention
        top_pairwise_rough_score, top_antecedent_ids, top_antecedent_mask, top_antecedent_offsets, max_top_antecedent=self.get_coarse_score(top_span_mention_scores, top_span_emb, num_top_spans, device)
        
        #get final corefered score for each remaining span pair
        top_pairwise_score=self.get_final_score(input_ids, input_mask, speaker_ids, genre_ids, top_span_starts, top_span_emb, top_antecedent_ids, top_antecedent_offsets, top_pairwise_rough_score, num_top_spans, max_top_antecedent, device)
        
        if(not do_loss): #Return the antecedent distribution for each candidate mention if no loss is computed
            top_antecedent_scores=torch.cat([torch.zeros((num_top_spans, 1), device=self.device), top_pairwise_score], dim=1)  #top_antecedent_scores(num_top_spans, max_top_antecedents+1), set 0 as the score for empty antecedent
            return candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedent_ids, top_antecedent_scores
        
        #Get gold labels
        top_antecedent_gold_labels=self.get_gold_labels(top_span_cluster_ids, top_antecedent_ids, top_antecedent_mask)
        
        #Get loss
        loss, top_antecedent_scores, log_marginalized_antecedent_scores, log_norm=self.get_loss(top_pairwise_score, top_antecedent_gold_labels, num_top_spans, device)

        return loss, [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedent_ids, top_antecedent_scores, gold_starts, gold_ends, gold_mention_cluster_ids]
    
    def get_predicted_antecedents(self, top_antecedent_ids, top_antecedent_scores):
        predicted_antecedents=[]
        for i, idx in enumerate(np.argmax(top_antecedent_scores, axis=1)-1):
            if idx<0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(top_antecedent_ids[i][idx])
        return predicted_antecedents
    
    def get_predicted_clusters(self, top_span_starts, top_span_ends, top_antecedent_ids, top_antecedent_scores):
        """top_span_starts,top_span_ends(num_top_spans)
           top_antecedent_ids(num_top_spans,max_top_antecedents)
           top_antecedent_scores(num_top_spans,max_top_antecedents+1)
        """
        predicted_antecedents=self.get_predicted_antecedents(top_antecedent_ids, top_antecedent_scores) #predicted_antecedents(num_top_spans)
        mention_to_cluster_id={} #record the cluster id for each mention
        predicted_clusters=[] #a list of predicted coref clusters
        for i, predicted_idx in enumerate(predicted_antecedents):
            if predicted_idx<0:
                continue
            #assert i>predicted_idx, f'span idx: {i}; antecedent idx: {predicted_idx};'
            # Check antecedent's cluster
            antecedent=(int(top_span_starts[predicted_idx]), int(top_span_ends[predicted_idx]))
            antecedent_cluster_id=mention_to_cluster_id.get(antecedent, -1)
            if antecedent_cluster_id==-1:
                antecedent_cluster_id=len(predicted_clusters)
                predicted_clusters.append([antecedent])
                mention_to_cluster_id[antecedent]=antecedent_cluster_id
            # Add mention to cluster
            mention=(int(top_span_starts[i]), int(top_span_ends[i]))
            predicted_clusters[antecedent_cluster_id].append(mention)
            mention_to_cluster_id[mention]=antecedent_cluster_id
            
        predicted_clusters=[tuple(c) for c in predicted_clusters]
        return predicted_clusters, mention_to_cluster_id, predicted_antecedents
        
    def get_gold_clusters(self, gold_starts, gold_ends, gold_mention_cluster_ids):
        cluster_id_to_mentions={} #record the list of mentions for each cluster id
        gold_clusters=[] #a list of gold coref clusters
        for i in range(len(gold_starts)):
            mention=(int(gold_starts[i]), int(gold_ends[i]))
            mention_id=gold_mention_cluster_ids[i]
            if mention_id in cluster_id_to_mentions:
                cluster_id_to_mentions[mention_id].append(mention)
            else:
                cluster_id_to_mentions[mention_id]=[mention]
        for k in cluster_id_to_mentions:
            gold_clusters.append(cluster_id_to_mentions[k])
            
        gold_clusters=[tuple(c) for c in gold_clusters]
        return gold_clusters, cluster_id_to_mentions
        
    def get_evaluation_results(self, top_span_starts, top_span_ends, top_antecedent_ids, top_antecedent_scores, gold_starts, gold_ends, gold_mention_cluster_ids):
        """CPU list input"""
        predicted_clusters, predicted_mention_to_cluster_id, _=self.get_predicted_clusters(top_span_starts, top_span_ends, top_antecedent_ids, top_antecedent_scores)
        mention_to_predicted = {m: predicted_clusters[cluster_idx] for m, cluster_idx in predicted_mention_to_cluster_id.items()} #dict mapping each mention to its predicted cluster of mentions
        gold_clusters, _=self.get_gold_clusters(gold_starts, gold_ends, gold_mention_cluster_ids)
        mention_to_gold={m: cluster for cluster in gold_clusters for m in cluster} #dict mapping each mention to its gold cluster of mentions
        return predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold