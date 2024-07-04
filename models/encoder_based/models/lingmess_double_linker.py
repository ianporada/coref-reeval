import sys
sys.path.append('/home/mila/x/xiyuan.zou/research/kd-coref')
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from transformers import AutoModel, BertPreTrainedModel, T5EncoderModel
from utilities.utils import mask_tensor, get_pronoun_id, get_category_id
from utilities.consts import STOPWORDS, CATEGORIES, NULL_ID_FOR_COREF

"""config: {"llm_pretrained_name":str, "max_span_len":int, "top_lambda":float, "ffnn_size":int, "dropout_prob":float}
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
    

class LingMess_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.max_span_len=config["max_span_len"]
        self.top_lambda=config["top_lambda"]
        self.ffnn_size=config["ffnn_size"]
        if config["llm_type"]!="t5":
            self.llm_encoder=AutoModel.from_pretrained(config["llm_pretrained_name"])
        else:
            self.llm_encoder=T5EncoderModel.from_pretrained(config["llm_pretrained_name"])
        self.encoder_emb_size=self.llm_encoder.config.hidden_size
        self.num_cats=len(CATEGORIES)+1
        self.all_cats_size=self.ffnn_size*self.num_cats
        
        self.start_mention_mlp=FullyConnectedLayer(self.encoder_emb_size, self.ffnn_size, config)
        self.end_mention_mlp=FullyConnectedLayer(self.encoder_emb_size, self.ffnn_size, config)
        self.mention_start_classifier=nn.Linear(self.ffnn_size, 1)
        self.mention_end_classifier=nn.Linear(self.ffnn_size, 1)
        self.mention_s2e_classifier=nn.Linear(self.ffnn_size, self.ffnn_size)
        
        self.coref_start_all_mlps=FullyConnectedLayer(self.encoder_emb_size, self.all_cats_size, config)
        self.coref_end_all_mlps=FullyConnectedLayer(self.encoder_emb_size, self.all_cats_size, config)

        self.antecedent_s2s_all_weights = nn.Parameter(torch.empty((self.num_cats, self.ffnn_size, self.ffnn_size)))
        self.antecedent_e2e_all_weights = nn.Parameter(torch.empty((self.num_cats, self.ffnn_size, self.ffnn_size)))
        self.antecedent_s2e_all_weights = nn.Parameter(torch.empty((self.num_cats, self.ffnn_size, self.ffnn_size)))
        self.antecedent_e2s_all_weights = nn.Parameter(torch.empty((self.num_cats, self.ffnn_size, self.ffnn_size)))

        self.antecedent_s2s_all_biases = nn.Parameter(torch.empty((self.num_cats, self.ffnn_size)))
        self.antecedent_e2e_all_biases = nn.Parameter(torch.empty((self.num_cats, self.ffnn_size)))
        self.antecedent_s2e_all_biases = nn.Parameter(torch.empty((self.num_cats, self.ffnn_size)))
        self.antecedent_e2s_all_biases = nn.Parameter(torch.empty((self.num_cats, self.ffnn_size)))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        W = [self.antecedent_s2s_all_weights, self.antecedent_e2e_all_weights,
             self.antecedent_s2e_all_weights, self.antecedent_e2s_all_weights]

        B = [self.antecedent_s2s_all_biases, self.antecedent_e2e_all_biases,
             self.antecedent_s2e_all_biases, self.antecedent_e2s_all_biases]

        for w, b in zip(W, B):
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(b, -bound, bound)
            
    def num_parameters(self):
        def head_filter(x):
            return x[1].requires_grad and any(hp in x[0] for hp in ['coref', 'mention', 'antecedent'])
        head_params = filter(head_filter, self.named_parameters())
        head_params = sum(p.numel() for n, p in head_params)
        return super().num_parameters() - head_params, head_params
        
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
    
    def get_mention_scores(self, start_mention_reps, end_mention_reps):
        start_mention_scores=self.mention_start_classifier(start_mention_reps).squeeze(-1)  #start_mention_scores(batch_size, seq_len)
        end_mention_scores=self.mention_end_classifier(end_mention_reps).squeeze(-1)  #end_mention_scores(batch_size, seq_len)
        temp=self.mention_s2e_classifier(start_mention_reps)  #temp(batch_size, seq_len, ffnn_size)
        joint_mention_scores=torch.matmul(temp, end_mention_reps.permute([0, 2, 1])) #joint_mention_scores(batch_size, seq_len, seq_len)
        mention_scores=joint_mention_scores+start_mention_scores.unsqueeze(-1)+end_mention_scores.unsqueeze(-2)
        mention_mask=torch.ones_like(mention_scores, device=self.device)
        mention_mask=torch.triu(mention_mask, diagonal=0)
        mention_mask=torch.tril(mention_mask, diagonal=self.max_span_len-1) #mention_mask(batch_size, seq_len, seq_len), 1 for valid spans, 0 for invalid spans
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
    
    def get_categories_labels(self, tokens, subtoken_map, new_token_map, span_starts, span_ends, batch_size, max_k):
        spans=[] #spans(batch_size, max_k), contains the tuple of literal mention words and its pronoun id for each top mention for each doc
        for b, (starts, ends) in enumerate(zip(span_starts.cpu().tolist(), span_ends.cpu().tolist())):
            doc_spans=[]
            for start, end in zip(starts, ends):
                token_indices=[new_token_map[b][idx] for idx in set(subtoken_map[b][start:end+1])-{None}] #which tokens each top mention map to 
                span={tokens[b][idx].lower() for idx in token_indices if idx is not None} #extract the literal form of the mention in the lowercase
                pronoun_id=get_pronoun_id(span) 
                doc_spans.append((span-STOPWORDS, pronoun_id))
            spans.append(doc_spans)

        categories_labels=np.zeros((batch_size, max_k, max_k))-1
        for b in range(batch_size):
            for i in range(max_k):
                for j in list(range(max_k))[:i]:
                    categories_labels[b, i, j]=get_category_id(spans[b][i], spans[b][j])

        categories_labels=torch.tensor(categories_labels, device=self.device) #categories_labels(batch_size, max_k, max_k), the category id of each pair of top mentions, -1 is invalid mention pair
        categories_masks=[categories_labels==cat_id for cat_id in range(self.num_cats-1)]+[categories_labels!=-1]
        categories_masks=torch.stack(categories_masks, dim=1).int()
        return categories_labels, categories_masks
    
    def transpose_for_scores(self, x):
        new_x_shape=x.size()[:-1]+(self.num_cats, self.ffnn_size)
        x=x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def get_coref_scores(self, start_reps, end_reps):
        all_starts=self.transpose_for_scores(self.coref_start_all_mlps(start_reps)) #all_starts, all_ends(batch_size,num_cats,max_k,ffnn_size)
        all_ends=self.transpose_for_scores(self.coref_end_all_mlps(end_reps))

        logits=torch.einsum('bnkf, nfg, bnlg -> bnkl', all_starts, self.antecedent_s2s_all_weights, all_starts)+\
                 torch.einsum('bnkf, nfg, bnlg -> bnkl', all_ends,   self.antecedent_e2e_all_weights, all_ends)+\
                 torch.einsum('bnkf, nfg, bnlg -> bnkl', all_starts, self.antecedent_s2e_all_weights, all_ends)+\
                 torch.einsum('bnkf, nfg, bnlg -> bnkl', all_ends,   self.antecedent_e2s_all_weights, all_starts)

        biases=torch.einsum('bnkf, nf -> bnk', all_starts, self.antecedent_s2s_all_biases).unsqueeze(-2)+\
                 torch.einsum('bnkf, nf -> bnk', all_ends,   self.antecedent_e2e_all_biases).unsqueeze(-2)+\
                 torch.einsum('bnkf, nf -> bnk', all_ends,   self.antecedent_s2e_all_biases).unsqueeze(-2)+\
                 torch.einsum('bnkf, nf -> bnk', all_starts, self.antecedent_e2s_all_biases).unsqueeze(-2)
        return logits+biases
    
    def mask_antecedent_logits(self, antecedent_logits, span_mask, categories_masks=None):
        antecedents_mask = torch.tril(torch.ones_like(antecedent_logits, device=self.device), diagonal=-1)
        if categories_masks is not None:
            mask = antecedents_mask * span_mask.unsqueeze(1).unsqueeze(-1)
            mask *= categories_masks
        else:
            mask = antecedents_mask * span_mask.unsqueeze(-1)
        antecedent_logits = mask_tensor(antecedent_logits, mask)
        return antecedent_logits
    
    def extract_clusters(self, gold_clusters): #remove the padded part of each cluster
        gold_clusters=[tuple(tuple(m) for m in gc if NULL_ID_FOR_COREF not in m) for gc in gold_clusters]
        gold_clusters=[cluster for cluster in gold_clusters if len(cluster)>0]
        return gold_clusters
    
    def extract_mentions_to_clusters(self, gold_clusters):
        mention_to_gold={}
        for gc in gold_clusters:
            for mention in gc:
                mention_to_gold[tuple(mention)]=gc
        return mention_to_gold
    
    def get_clusters_labels(self, span_starts, span_ends, all_clusters):
        """
        :param span_starts: [batch_size, max_k]
        :param span_ends: [batch_size, max_k]
        :param all_clusters: [batch_size, max_cluster_size, max_clusters_num, 2]
        :return: [batch_size, max_k, max_k + 1] - [b, i, j] == 1 if j is antecedent of i
        """
        batch_size, max_k = span_starts.size()
        new_cluster_labels = np.zeros((batch_size, max_k, max_k))
        span_starts_cpu = span_starts.cpu().tolist()
        span_ends_cpu = span_ends.cpu().tolist()
        all_clusters_cpu = all_clusters.cpu().tolist()
        for b, (starts, ends, gold_clusters) in enumerate(zip(span_starts_cpu, span_ends_cpu, all_clusters_cpu)):
            gold_clusters = self.extract_clusters(gold_clusters)
            mention_to_gold_clusters = self.extract_mentions_to_clusters(gold_clusters)
            for i, (start, end) in enumerate(zip(starts, ends)):
                if (start, end) not in mention_to_gold_clusters:
                    continue
                for j, (a_start, a_end) in enumerate(list(zip(starts, ends))[:i]):
                    if (a_start, a_end) in mention_to_gold_clusters[(start, end)]:
                        new_cluster_labels[b, i, j] = 1
        new_cluster_labels = torch.tensor(new_cluster_labels, device=self.device)
        return new_cluster_labels
    
    def get_all_labels(self, clusters_labels, categories_masks):
        batch_size, max_k, _ = clusters_labels.size()
        categories_labels = clusters_labels.unsqueeze(1).repeat(1, self.num_cats, 1, 1) * categories_masks
        all_labels = torch.cat((categories_labels, clusters_labels.unsqueeze(1)), dim=1)            # for the combined loss (L_coref + L_tasks)
        # null cluster
        zeros = torch.zeros((batch_size, self.num_cats + 1, max_k, 1), device=self.device)
        all_labels = torch.cat((all_labels, zeros), dim=-1)                                         # [batch_size, num_cats + 1, max_k, max_k + 1]
        no_antecedents = 1 - torch.sum(all_labels, dim=-1).bool().float()
        all_labels[:, :, :, -1] = no_antecedents
        return all_labels
    
    def get_marginal_log_likelihood_loss(self, logits, labels, span_mask):
        gold_coref_logits = mask_tensor(logits, labels)                       # [batch_size, num_cats + 1, max_k, max_k]
        gold_log_sum_exp = torch.logsumexp(gold_coref_logits, dim=-1)         # [batch_size, num_cats + 1, max_k]
        all_log_sum_exp = torch.logsumexp(logits, dim=-1)                     # [batch_size, num_cats + 1, max_k]
        losses = all_log_sum_exp - gold_log_sum_exp                           # [batch_size, num_cats + 1, max_k]
        # zero the loss of padded spans
        span_mask = span_mask.unsqueeze(1)                                    # [batch_size, 1, max_k]
        losses = losses * span_mask                                           # [batch_size, num_cats, max_k]
        # normalize loss by spans
        per_span_loss = losses.mean(dim=-1)                                   # [batch_size, num_cats + 1]
        # normalize loss by document
        loss_per_cat = per_span_loss.mean(dim=0)                              # [num_cats + 1]
        # normalize loss by category
        loss = loss_per_cat.sum()
        return loss
    
    def forward(self, input_ids, input_masks, gold_clusters, tokens, subtoken_map, new_token_map, leftover_input_ids=None, leftover_input_masks=None):
        """input_ids(batch_size, seq_len), input_masks(batch_size, seq_len), gold_clusters(batch_size, num_clusters)
           tokens(batch_size, num_words), a list of words for each doc on cpu
           subtoken_map(batch_size, seq_len), a list of token number of each subtoken for each doc on cpu 
           new_token_map(batch_size, num_tokens), a list of word number of each token for each doc on cpu 
        """
        self.device=input_ids.device
        
        if leftover_input_ids==None:
            input_emb=self.get_input_emb(input_ids, input_masks) #input_emb(batch_size, seq_len, encoder_emb_size)
        else:
            input_emb, input_masks=self.get_input_emb_leftover(input_ids, input_masks, leftover_input_ids, leftover_input_masks)
        start_mention_reps=self.start_mention_mlp(input_emb) #start_mention_reps, end_mention_reps(batch_size,seq_len,ffnn_size)
        end_mention_reps=self.end_mention_mlp(input_emb)
        mention_scores=self.get_mention_scores(start_mention_reps, end_mention_reps) #mention_scores(batch_size, seq_len, seq_len)
        topk_mention_start_ids, topk_mention_end_ids, span_mask, topk_mention_logits, batch_size, max_k=self.get_topk_mentions(mention_scores, input_masks) 
        categories_labels, categories_masks=self.get_categories_labels(tokens, subtoken_map, new_token_map, topk_mention_start_ids, topk_mention_end_ids, batch_size, max_k)
        size=(batch_size, max_k, self.encoder_emb_size)
        topk_start_reps=torch.gather(input_emb, dim=1, index=topk_mention_start_ids.unsqueeze(-1).expand(size)) #topk_start_reps, topk_end_reps(batch_size,max_k,llm_emb_size) 
        topk_end_reps=torch.gather(input_emb, dim=1, index=topk_mention_end_ids.unsqueeze(-1).expand(size))
        categories_logits=self.get_coref_scores(topk_start_reps, topk_end_reps)
        
        final_logits=categories_logits*categories_masks
        final_logits=final_logits.sum(dim=1)+topk_mention_logits
        categories_logits=categories_logits+topk_mention_logits.unsqueeze(1)
        # lower logits of padded spans or different category.
        final_logits=self.mask_antecedent_logits(final_logits, span_mask)
        categories_logits=self.mask_antecedent_logits(categories_logits, span_mask, categories_masks)
        # adding zero logits for null span
        final_logits=torch.cat((final_logits, torch.zeros((batch_size, max_k, 1), device=self.device)), dim=-1)                           # [batch_size, max_k, max_k + 1]
        categories_logits=torch.cat((categories_logits, torch.zeros((batch_size, self.num_cats, max_k, 1), device=self.device)), dim=-1)  # [batch_size, num_cats, max_k, max_k + 1]

        if gold_clusters is not None:
            clusters_labels=self.get_clusters_labels(topk_mention_start_ids, topk_mention_end_ids, gold_clusters)
            all_labels=self.get_all_labels(clusters_labels, categories_masks)
            all_logits=torch.cat((categories_logits, final_logits.unsqueeze(1)), dim=1)
            loss=self.get_marginal_log_likelihood_loss(all_logits, all_labels, span_mask)
        
        return loss, [topk_mention_start_ids, topk_mention_end_ids, final_logits, gold_clusters]

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
        mention_to_predicted=self.extract_mentions_to_clusters(predicted_clusters)
        gold_clusters=self.extract_clusters(gold_clusters)
        mention_to_gold=self.extract_mentions_to_clusters(gold_clusters)
        return predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold
        