import sys
sys.path.append('/home/mila/x/xiyuan.zou/research/kd-coref')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from transformers import AutoModel, T5EncoderModel
from utilities.utils import mask_tensor
from utilities.consts import NULL_ID_FOR_COREF

"""config: {"llm_pretrained_name":str, "max_span_len":int, "top_lambda":float, "ffnn_size_mention_detector":int, "ffnn_size_mention_linker":int, "dropout_prob":float, "normalise_loss":Bool}
"""
class FullyConnectedLayer(nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.dropout_prob=config["dropout_prob"]
        
        self.dense=nn.Linear(self.input_dim, self.output_dim)
        self.layer_norm=nn.LayerNorm(self.output_dim)
        self.activation_func=nn.GELU()
        self.dropout=nn.Dropout(self.dropout_prob)

    def forward(self, inputs):
        temp=inputs
        temp=self.dense(temp)
        temp=self.activation_func(temp)
        temp=self.layer_norm(temp)
        temp=self.dropout(temp)
        return temp

class S2E_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.max_span_len=config["max_span_len"]
        self.top_lambda=config["top_lambda"]
        self.ffnn_size_d=config["ffnn_size_mention_detector"]
        self.ffnn_size_l=config["ffnn_size_mention_linker"]
        self.normalise_loss=config["normalise_loss"]
        
        if config["llm_type"]!="t5":
            self.llm_encoder=AutoModel.from_pretrained(config["llm_pretrained_name"])
        else:
            self.llm_encoder=T5EncoderModel.from_pretrained(config["llm_pretrained_name"])
        self.encoder_emb_size=self.llm_encoder.config.hidden_size
        
        self.start_mention_mlp=FullyConnectedLayer(self.encoder_emb_size, self.ffnn_size_d, config)
        self.end_mention_mlp=FullyConnectedLayer(self.encoder_emb_size, self.ffnn_size_d, config)
        self.start_coref_mlp=FullyConnectedLayer(self.encoder_emb_size, self.ffnn_size_l, config)
        self.end_coref_mlp=FullyConnectedLayer(self.encoder_emb_size, self.ffnn_size_l, config)
        self.mention_start_classifier=nn.Linear(self.ffnn_size_d, 1)
        self.mention_end_classifier=nn.Linear(self.ffnn_size_d, 1)
        self.mention_s2e_classifier=nn.Linear(self.ffnn_size_d, self.ffnn_size_d)
        self.antecedent_s2s_classifier=nn.Linear(self.ffnn_size_l, self.ffnn_size_l)
        self.antecedent_e2e_classifier=nn.Linear(self.ffnn_size_l, self.ffnn_size_l)
        self.antecedent_s2e_classifier=nn.Linear(self.ffnn_size_l, self.ffnn_size_l)
        self.antecedent_e2s_classifier=nn.Linear(self.ffnn_size_l, self.ffnn_size_l)
    
    def get_input_emb(self, input_ids, input_masks):
        output_dict=self.llm_encoder(input_ids, attention_mask=input_masks, return_dict=True)
        input_emb=output_dict["last_hidden_state"]
        return input_emb
    
    def get_input_emb_leftover(self, input_ids, input_masks, leftover_input_ids, leftover_input_masks):
        docs, segments, segment_len=input_ids.size()
        input_ids, input_masks=input_ids.view(-1, segment_len), input_masks.view(-1, segment_len)
        input_emb=self.get_input_emb(input_ids, input_masks)
        input_masks=input_masks.view((docs, segments*segment_len))  
        input_emb=input_emb.view((docs, segments*segment_len, -1))  
        
        leftover_input_emb=self.get_input_emb(leftover_input_ids, leftover_input_masks)
        input_emb=torch.cat([input_emb, leftover_input_emb], dim=1) #input_masks(batch_size, seq_len)
        input_masks=torch.cat([input_masks, leftover_input_masks], dim=1) #input_emb(batch_size, seq_len, encoder_emb_size)
        return input_emb, input_masks
    
    def compute_start_end_representations(self, input_emb):
        start_mention_reps=self.start_mention_mlp(input_emb)
        end_mention_reps=self.end_mention_mlp(input_emb)
        start_coref_reps=self.start_coref_mlp(input_emb)
        end_coref_reps=self.end_coref_mlp(input_emb)
        return start_mention_reps, end_mention_reps, start_coref_reps, end_coref_reps
    
    def get_mention_scores(self, start_mention_reps, end_mention_reps):
        start_mention_scores=self.mention_start_classifier(start_mention_reps).squeeze(-1)  #start_mention_scores(batch_size, seq_len)
        end_mention_scores=self.mention_end_classifier(end_mention_reps).squeeze(-1)  #end_mention_scores(batch_size, seq_len)
        temp=self.mention_s2e_classifier(start_mention_reps)  #temp(batch_size, seq_len, ffnn_size)
        joint_mention_scores=torch.matmul(temp, end_mention_reps.permute([0, 2, 1])) #joint_mention_scores(batch_size, seq_len, seq_len)
        mention_scores=joint_mention_scores+start_mention_scores.unsqueeze(-1)+end_mention_scores.unsqueeze(-2)
        mention_mask=torch.ones_like(mention_scores, device=self.device, dtype=torch.float)
        mention_mask=mention_mask.triu(diagonal=0)
        mention_mask=mention_mask.tril(diagonal=self.max_span_len-1) #mention_mask(batch_size, seq_len, seq_len), 1 for valid spans, 0 for invalid spans
        mention_scores=mask_tensor(mention_scores, mention_mask) #mention_scores(batch_size, seq_len, seq_len)，mention score calculated based only on the start and end token for every span
        return mention_scores
    
    def get_topk_mentions(self, mention_scores, input_masks):
        batch_size, seq_length, _=mention_scores.shape
        actual_seq_lengths=torch.sum(input_masks, dim=-1)  #the actual token length of each doc(batch_size)
        k=(actual_seq_lengths*self.top_lambda).int()  # k (batch_size), number of mentions kept for each doc
        max_k=int(torch.max(k)) #need to pad to max_k later
        
        _, topk_1d_indices=torch.topk(mention_scores.view(batch_size, -1), dim=-1, k=max_k) #topk_1d_indices(batch_size, max_k), the span id of each kept mention for each doc
        size=(batch_size, max_k)
        idx=torch.arange(max_k, device=self.device).unsqueeze(0).expand(size)
        len_expanded=k.unsqueeze(1).expand(size)
        span_mask=(idx<len_expanded).int() #span_mask(batch_size, max_k), 1 for valid kept spans, 0 for padded spans
        topk_1d_indices=(topk_1d_indices*span_mask)+(1-span_mask)*((seq_length**2)-1) #Take different k values for each doc
        sorted_topk_1d_indices, _ =torch.sort(topk_1d_indices, dim=-1)
        topk_mention_start_ids=sorted_topk_1d_indices//seq_length #topk_mention_start_ids, topk_mention_end_ids(batch_size, max_k), the start token id and end token id of each kept mention for each doc
        topk_mention_end_ids=sorted_topk_1d_indices%seq_length
        topk_mention_scores=mention_scores[torch.arange(batch_size).unsqueeze(-1).expand(batch_size, max_k), topk_mention_start_ids, topk_mention_end_ids]  # [batch_size, max_k]
        topk_mention_scores=topk_mention_scores.unsqueeze(-1)+topk_mention_scores.unsqueeze(-2)  #topk_mention_scores(batch_size, max_k, max_k), the mention score of each kept span plus the mention score of each kept span in the same doc for each doc
        return topk_mention_start_ids, topk_mention_end_ids, span_mask, topk_mention_scores, batch_size, max_k
        
    def get_coref_scores(self, start_coref_reps, end_coref_reps, topk_mention_start_ids, topk_mention_end_ids, batch_size, max_k):
        size=(batch_size, max_k, start_coref_reps.shape[-1])
        top_k_start_coref_reps=torch.gather(start_coref_reps, dim=1, index=topk_mention_start_ids.unsqueeze(-1).expand(size)) #topk_start_coref_reps(batch_size,max_k,ffnn_size), for each doc, start token reps of each kept span used for coref
        top_k_end_coref_reps=torch.gather(end_coref_reps, dim=1, index=topk_mention_end_ids.unsqueeze(-1).expand(size)) #topk_end_coref_reps(batch_size,max_k,ffnn_size), for each doc, end token reps of each kept span used for coref
        
        temp=self.antecedent_s2s_classifier(top_k_start_coref_reps) #(batch_size, max_k, ffnn_size)
        top_k_s2s_coref_scores=torch.matmul(temp, top_k_start_coref_reps.permute([0, 2, 1])) #(batch_size, max_k, max_k)
        temp=self.antecedent_e2e_classifier(top_k_end_coref_reps)   #(batch_size, max_k, ffnn_size)
        top_k_e2e_coref_scores=torch.matmul(temp, top_k_end_coref_reps.permute([0, 2, 1]))  #(batch_size, max_k, max_k)
        temp=self.antecedent_s2e_classifier(top_k_start_coref_reps) #(batch_size, max_k, ffnn_size)
        top_k_s2e_coref_scores=torch.matmul(temp, top_k_end_coref_reps.permute([0, 2, 1]))  #(batch_size, max_k, max_k)
        temp=self.antecedent_e2s_classifier(top_k_end_coref_reps)  #(batch_size, max_k, ffnn_size)
        top_k_e2s_coref_scores=torch.matmul(temp, top_k_start_coref_reps.permute([0, 2, 1])) #(batch_size, max_k, max_k)
        
        coref_scores=top_k_s2e_coref_scores+top_k_e2s_coref_scores+top_k_s2s_coref_scores+top_k_e2e_coref_scores
        return coref_scores #coref_scores(batch_size, max_k, max_k), the coref scores between each kept span and each kept span in the same doc for each doc
        
    def finalize_coref_scores(self, topk_mention_scores, coref_scores, span_mask, batch_size, max_k):
        final_scores=topk_mention_scores+coref_scores
        antecedents_mask=torch.ones_like(final_scores).tril(diagonal=-1)  
        antecedents_mask=antecedents_mask*span_mask.unsqueeze(-1)  
        final_scores=mask_tensor(final_scores, antecedents_mask) #final_scores(batch_size, max_k, max_k)
        final_scores=torch.cat((final_scores, torch.zeros((batch_size, max_k, 1), device=self.device)), dim=-1) #final_scores(batch_size, max_k, max_k+1), include the empty span for each span 
        return final_scores
        
    def get_cluster_labels_after_pruning(self, span_starts, span_ends, all_clusters, batch_size, max_k):
        new_cluster_labels=torch.zeros((batch_size, max_k, max_k+1), device='cpu') #new_cluster_labels(batch_size, max_k, max_k+1), [b,i,j]=1 if span i and j corefer otherwise 0
        all_clusters_cpu=all_clusters.cpu().numpy()
        for b, (starts, ends, gold_clusters) in enumerate(zip(span_starts.cpu().tolist(), span_ends.cpu().tolist(), all_clusters_cpu)):
            gold_clusters=self.extract_clusters(gold_clusters)
            mention_to_gold_clusters=self.extract_mentions_to_predicted_clusters_from_clusters(gold_clusters)
            gold_mentions=set(mention_to_gold_clusters.keys())
            for i, (start, end) in enumerate(zip(starts, ends)):
                if (start, end) not in gold_mentions:
                    continue
                for j, (a_start, a_end) in enumerate(list(zip(starts, ends))[:i]):
                    if (a_start, a_end) in mention_to_gold_clusters[(start, end)]:
                        new_cluster_labels[b, i, j]=1
        new_cluster_labels=new_cluster_labels.to(self.device)
        no_antecedents=1-torch.sum(new_cluster_labels, dim=-1).bool().float()
        new_cluster_labels[:, :, -1]=no_antecedents
        return new_cluster_labels
    
    def get_loss(self, final_scores, labels_after_pruning, span_mask):
        gold_coref_scores=mask_tensor(final_scores, labels_after_pruning)
        gold_log_sum_exp=torch.logsumexp(gold_coref_scores, dim=-1)  
        all_log_sum_exp=torch.logsumexp(final_scores, dim=-1)  
        gold_log_probs=gold_log_sum_exp-all_log_sum_exp
        losses=-gold_log_probs
        losses=losses*span_mask
        per_example_loss=torch.sum(losses, dim=-1)  # [batch_size]
        if self.normalise_loss:
            per_example_loss=per_example_loss/losses.shape[-1]
        loss=per_example_loss.mean()
        return loss
    
    def forward(self, input_ids, input_masks, gold_clusters=None, leftover_input_ids=None, leftover_input_masks=None):
        """input_ids(batch_size, seq_len), input_masks(batch_size, seq_len)
           gold_clusters(batch_size, num_clusters)
        """
        self.device=input_ids.device
        if leftover_input_ids==None:
            input_emb=self.get_input_emb(input_ids, input_masks) #input_emb(batch_size, seq_len, encoder_emb_size)
        else:
            input_emb, input_masks=self.get_input_emb_leftover(input_ids, input_masks, leftover_input_ids, leftover_input_masks)
        start_mention_reps, end_mention_reps, start_coref_reps, end_coref_reps=self.compute_start_end_representations(input_emb) #start_mention_reps, end_mention_reps, start_coref_reps, end_coref_reps(batch_size,seq_len,ffnn_size)
        mention_scores=self.get_mention_scores(start_mention_reps, end_mention_reps) #mention_scores(batch_size, seq_len, seq_len)
        topk_mention_start_ids, topk_mention_end_ids, span_mask, topk_mention_scores, batch_size, max_k=self.get_topk_mentions(mention_scores, input_masks) 
        coref_scores=self.get_coref_scores(start_coref_reps, end_coref_reps, topk_mention_start_ids, topk_mention_end_ids, batch_size, max_k)
        final_scores=self.finalize_coref_scores(topk_mention_scores, coref_scores, span_mask, batch_size, max_k) #final_scores(batch_size, max_k, max_k+1)
        
        loss=None
        if gold_clusters is not None: #need to do loss
            labels_after_pruning=self.get_cluster_labels_after_pruning(topk_mention_start_ids, topk_mention_end_ids, gold_clusters, batch_size, max_k)
            loss=self.get_loss(final_scores, labels_after_pruning, span_mask)
        return loss, [topk_mention_start_ids, topk_mention_end_ids, final_scores, gold_clusters]
    
    def extract_clusters(self, gold_clusters): #remove the padded part of each cluster
        gold_clusters=[tuple(tuple(m) for m in gc if NULL_ID_FOR_COREF not in m) for gc in gold_clusters.tolist()]
        gold_clusters=[cluster for cluster in gold_clusters if len(cluster)>0]
        return gold_clusters
    
    def extract_mentions_to_predicted_clusters_from_clusters(self, gold_clusters):
        mention_to_gold={}
        for gc in gold_clusters:
            for mention in gc:
                mention_to_gold[tuple(mention)]=gc
        return mention_to_gold
    
    def extract_clusters_for_decode(self, mention_to_antecedent):
        mention_to_antecedent=sorted(mention_to_antecedent)
        mention_to_cluster={}
        clusters=[]
        for mention, antecedent in mention_to_antecedent:
            if antecedent in mention_to_cluster:
                cluster_idx=mention_to_cluster[antecedent]
                clusters[cluster_idx].append(mention)
                mention_to_cluster[mention]=cluster_idx
            else:
                cluster_idx=len(clusters)
                mention_to_cluster[mention]=cluster_idx
                mention_to_cluster[antecedent]=cluster_idx
                clusters.append([antecedent, mention])
        clusters=[tuple(cluster) for cluster in clusters]
        return clusters, mention_to_cluster
    
    def get_evaluation_results(self, start_ids, end_ids, final_scores, gold_clusters):
        """all inputs are numpy arrays on cpu
           start_ids(max_k), end_ids(max_k), final_scores(max_k,max_k+1) 
        """
        max_antecedents=np.argmax(final_scores, axis=1).tolist() #max_antecedents(max_k), antecedent for each top span
        mention_to_antecedent={((int(start), int(end)), (int(start_ids[max_antecedent]), int(end_ids[max_antecedent]))) for start, end, max_antecedent in zip(start_ids, end_ids, max_antecedents) if max_antecedent<len(start_ids)}
        predicted_clusters, _=self.extract_clusters_for_decode(mention_to_antecedent)
        mention_to_predicted=self.extract_mentions_to_predicted_clusters_from_clusters(predicted_clusters)
        gold_clusters=self.extract_clusters(gold_clusters)
        mention_to_gold=self.extract_mentions_to_predicted_clusters_from_clusters(gold_clusters)
        return predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold
