import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.init as init
from transformers import BertModel
import sys
sys.path.append("src")
from utils import add_dummy

class SpanPredictor(nn.Module):
    def __init__(self, input_size, distance_emb_size):
        super().__init__()
        self.ffnn=torch.nn.Sequential(
            torch.nn.Linear(input_size*2+64, input_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 64),
        )
        self.conv=torch.nn.Sequential(
            torch.nn.Conv1d(64, 4, 3, 1, 1),
            torch.nn.Conv1d(4, 2, 3, 1, 1)
        )
        self.emb=torch.nn.Embedding(128, distance_emb_size)
        
    def get_dist_emb(self, token_emb, heads_ids):
        relative_positions=(heads_ids.unsqueeze(1)-torch.arange(token_emb.shape[0], device=token_emb.device).unsqueeze(0))
        emb_ids=relative_positions+63 # make all valid distances positive
        emb_ids[(emb_ids<0)+(emb_ids>126)]=127  # "too_far"
        return emb_ids, relative_positions #emb_ids(num_heads, num_tokens)
    
    def get_pair_emb(self, token_emb, heads_ids, sent_ids, emb_ids):
        same_sent_mask=(sent_ids[heads_ids].unsqueeze(1)==sent_ids.unsqueeze(0)) #same_sent_mask(num_heads, num_tokens)，if each head and each token are in the same sentence
        #Assume k is the number of head-token pair that the head and the token in the pair are in the same sentence
        rows, cols=same_sent_mask.nonzero(as_tuple=True) #rows, cols(k), the head index and token index for every same-sentence head-token pair
        #pair_matrix(k, bert_emb_size*2+distance_emb_size), for each head, concatenate each token in the same sentence with this head to get head-token pairs
        #the embedding for each such head-token pair consists of [head_emb, token_emb, distance_emb]
        pair_matrix=torch.cat((token_emb[heads_ids[rows]], token_emb[cols], self.emb(emb_ids[rows, cols])), dim=1) 
        lengths=same_sent_mask.sum(dim=1) #lengths(num_heads), the number of tokens in the same sentence as each head
        padding_mask=torch.arange(0, lengths.max(), device=token_emb.device).unsqueeze(0)
        padding_mask=(padding_mask<lengths.unsqueeze(1)) #padding_mask(num_heads, max_sent_len), for each head, 1 at position within its sentence length, 0 at position outside its sentence length
        pair_emb=torch.zeros(*padding_mask.shape, pair_matrix.shape[-1], device=token_emb.device)
        pair_emb[padding_mask]=pair_matrix #pair_emb(num_heads, max_sent_len, input_size*2+distance_emb_size), the embedding for each same-sentence head-token pair, padded to the max sentence length in the doc 
        return pair_emb, padding_mask, rows, cols
    
    def get_start_end_score(self, heads_ids, token_emb, pair_emb, padding_mask, rows, cols, relative_positions, training):
        res=self.ffnn(pair_emb) 
        res=self.conv(res.permute(0, 2, 1)).permute(0, 2, 1) #res(num_heads, max_sent_len, 2)
        res=res.to(torch.float32)
        scores=torch.full((heads_ids.shape[0], token_emb.shape[0], 2), float('-inf'), device=token_emb.device)
        scores[rows, cols]=res[padding_mask] #scores(num_heads, num_tokens, 2), for each head, a start score and an end score for each token within the same sentence, -inf for other tokens
        if not training: #Make sure that start <= head <= end during inference
            valid_starts=torch.log((relative_positions>=0).to(torch.float))
            valid_ends=torch.log((relative_positions<=0).to(torch.float))
            valid_positions=torch.stack((valid_starts, valid_ends), dim=2)
            scores+=valid_positions
        return scores
    
    def forward(self, token_emb, heads_ids, sent_ids, training=True):
        """token_emb(num_tokens, bert_emb_size): contextual embeddings for each token
           heads_ids(num_heads): token index of each span head
           sent_ids(num_tokens): the sentence id for each token
        """
        # get the distance between each head and each token for distance embeddings
        emb_ids, relative_positions=self.get_dist_emb(token_emb, heads_ids)
        
        #get the embedding for each same-sentence head-token pair
        pair_emb, padding_mask, rows, cols=self.get_pair_emb(token_emb, heads_ids, sent_ids, emb_ids)

        #get the start and end score of each token within the same sentence for each head
        scores=self.get_start_end_score(heads_ids, token_emb, pair_emb, padding_mask, rows, cols, relative_positions, training)
        
        return scores
    
    
    
class CorefLoss(torch.nn.Module):
    def __init__(self, bce_weight):
        assert 0<=bce_weight<=1
        super().__init__()
        self._bce_module=nn.BCEWithLogitsLoss()
        self._bce_weight=bce_weight

    def forward(self, input_, target):
        """ Returns a weighted sum of two losses """
        return (self._nlml(input_, target)+self._bce(input_, target)*self._bce_weight)

    def _bce(self, input_, target):
        return self._bce_module(torch.clamp(input_, min=-50, max=50), target) #For numerical stability, clamps the input before passing it to BCE

    @staticmethod
    def _nlml(input_, target):
        gold=torch.logsumexp(input_+torch.log(target), dim=1)
        input_=torch.logsumexp(input_, dim=1)
        return (input_-gold).mean()  
    
    
"""config: {"bert_pretrained_name":str, "dropout_rate":float, "max_num_candidate_antecedents":int, "feature_emb_size":int,
            "num_genres":int, "fine_score_batch_size":int, "ffnn_depth":int, "ffnn_size":int, "span_predictor_emb_size":int,
            "bce_loss_weight":float, "max_segment_len":int}
   max_num_candidate_antecedents: how many antecedent tokens with highest rough scores are kept for each token
   fine_score_batch_size: computation of fine corefer scores is memory intensive, need to split a full doc into batches
   span_predictor_emb_size：the embedding size used for the span predictor
"""
class BertWL_Model(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config=config
        self.device=device
        self.k=config["max_num_candidate_antecedents"]
        self.feature_emb_size=config["feature_emb_size"]
        self.num_genres=config["num_genres"]
        self.fine_score_batch_size=config["fine_score_batch_size"]
        self.span_predictor_emb_size=config["span_predictor_emb_size"]
        self.bce_loss_weight=config["bce_loss_weight"]
        
        self.bert=BertModel.from_pretrained(self.config["bert_pretrained_name"])
        self.bert_emb_size=self.bert.config.hidden_size
        self.dropout=nn.Dropout(p=self.config["dropout_rate"])
        self.token_attn=nn.Linear(in_features=self.bert_emb_size, out_features=1)
        self.rough_bilinear=nn.Linear(in_features=self.bert_emb_size, out_features=self.bert_emb_size)
        self.genre_emb=nn.Embedding(self.num_genres, self.feature_emb_size)
        self.distance_emb=nn.Embedding(9, self.feature_emb_size)
        self.speaker_emb=nn.Embedding(2, self.feature_emb_size)
        self.pair_emb_size=self.bert_emb_size*3+self.feature_emb_size*3
        self.span_predictor=SpanPredictor(self.bert_emb_size, self.span_predictor_emb_size)
        
        layers=[]
        ffnn_depth=self.config["ffnn_depth"]
        ffnn_size=self.config["ffnn_size"]
        for i in range(ffnn_depth):
            layers.extend([nn.Linear(ffnn_size if i else self.pair_emb_size, ffnn_size), nn.LeakyReLU(), nn.Dropout(self.config["dropout_rate"])])
        layers.append(nn.Linear(ffnn_size, out_features=1))
        self.fine_score_ffnn=nn.Sequential(*layers)
        
        self.coref_loss=CorefLoss(self.bce_loss_weight)
        self.span_loss=nn.CrossEntropyLoss(reduction="sum")
        
    def get_subtoken_emb(self, input_ids, input_mask):
        subtoken_emb,_=self.bert(input_ids, attention_mask=input_mask, return_dict=False) #subtoken_emb(num_segments, max_seg_len, bert_emb_size)
        input_mask=input_mask.to(torch.bool)
        subtoken_emb=subtoken_emb[input_mask] #subtoken_emb(num_subtokens, bert_emb_size)
        return subtoken_emb
    
    def get_token_emb(self, subtoken_emb, word2subword):
        token_starts=word2subword[:,0] #(num_tokens)
        token_ends=word2subword[:,1]
        num_tokens, num_subtokens=len(token_starts), len(subtoken_emb)
        token_attn_mask=torch.arange(0, num_subtokens, device=self.device).expand((num_tokens, num_subtokens))
        token_attn_mask=((token_attn_mask>=token_starts.unsqueeze(1))*(token_attn_mask<=token_ends.unsqueeze(1)))
        token_attn_mask=torch.log(token_attn_mask.to(torch.float)) #token_attn_mask(num_tokens, num_subtokens), for each token, 0 at its compositional subtokens, -inf at other subtokens
        token_attn_scores=self.token_attn(subtoken_emb).T
        token_attn_scores=token_attn_scores.expand((num_tokens, num_subtokens))+token_attn_mask 
        token_attn_weights=torch.softmax(token_attn_scores, dim=1) #token_attn_weights(num_tokens, num_subtokens), for each token, non-zero at its compositional subtokens, 0 at other subtokens
        token_emb=torch.mm(token_attn_weights, subtoken_emb)
        token_emb=self.dropout(token_emb) #token_emb(num_tokens, bert_emb_size)
        return token_emb, num_tokens, num_subtokens, token_attn_weights
    
    def get_rough_score(self, token_emb, num_tokens):
        pair_mask=torch.arange(0, num_tokens, device=self.device)
        pair_mask=pair_mask.unsqueeze(1)-pair_mask.unsqueeze(0)
        pair_mask=torch.log((pair_mask>0).to(torch.float)) #pair_mask(num_tokens, num_tokens), for each token, 0 for any previous tokens, -inf for all other tokens
        rough_scores=self.dropout(torch.mm(self.rough_bilinear(token_emb), token_emb.T))+pair_mask #rough_scores(num_tokens, num_tokens), rough corefer score between each token and all its previous tokens, -inf for other tokens
        top_scores, top_indices=torch.topk(rough_scores, k=min(self.k, len(rough_scores)), dim=1, sorted=False)
        return top_scores, top_indices #top_scores, top_indices(num_tokens, k), the indices and rough corefer scores between each token and its top k antecedents
    
    def get_pairwise_features(self, top_indices, speaker_ids, genre_id, num_tokens):
        same_speaker=(speaker_ids[top_indices]==speaker_ids.unsqueeze(1)) #same_speaker(num_tokens, k), 1 if token-antecedent pair has the same speaker, 0 otherwise
        same_speaker_emb=self.speaker_emb(same_speaker.to(torch.long)) #same_speaker_emb(num_tokens, k, feature_emb_size)
        token_ids=torch.arange(0, num_tokens, device=self.device)
        distance=(token_ids.unsqueeze(1)-token_ids[top_indices]).clamp_min_(min=1)
        log_distance=distance.to(torch.float).log2().floor_()
        log_distance=log_distance.clamp_max_(max=6).to(torch.long)
        distance=torch.where(distance<5, distance-1, log_distance+2)
        token_distance_emb=self.distance_emb(distance) #token_distance_emb(num_tokens, k, feature_emb_size)
        genre_id=genre_id.expand_as(top_indices)
        doc_genre_emb=self.genre_emb(genre_id) #doc_genre_emb(num_tokens, k, feature_emb_size)
        feature_emb=self.dropout(torch.cat((same_speaker_emb, token_distance_emb, doc_genre_emb), dim=2)) #feature_emb(num_tokens, k, feature_emb_size*3)
        return feature_emb
    
    def get_corefer_score_batch(self, token_emb, token_emb_batch, top_pairwise_features_batch, top_rough_scores_batch, top_indices_batch):
        source_emb=token_emb_batch.unsqueeze(1).expand(-1, top_pairwise_features_batch.shape[1], self.bert_emb_size) #source_emb(batch_size, k, bert_emb_size)
        antecedent_emb=token_emb[top_indices_batch] #antecedent_emb(batch_size, k, bert_emb_size)
        similarity=source_emb*antecedent_emb
        pair_emb=torch.cat((source_emb, antecedent_emb, similarity, top_pairwise_features_batch), dim=2) #pair_emb(batch_size, k, bert_emb_size*3+feature_emb_size*3)
        top_fine_scores_batch=self.fine_score_ffnn(pair_emb) #top_fine_scores_batch(batch_size, k)
        top_corefer_scores_batch=top_rough_scores_batch+top_fine_scores_batch.squeeze(2)
        top_corefer_scores_batch=add_dummy(top_corefer_scores_batch, eps=1e-7) #top_corefer_scores_batch(batch_size, k+1), score eps for empty antecedent
        return top_corefer_scores_batch
    
    def get_corefer_score(self, token_emb, top_indices, top_rough_scores, top_pairwise_features, num_tokens):
        corefer_scores_list=[]
        for i in range(0, num_tokens, self.fine_score_batch_size):
            #top_pairwise_features_batch(batch_size, k, feature_emb_size*3), token_emb_batch(batch_size, bert_emb_size)
            #top_indices_batch, top_rough_scores_batch(batch_size, k)
            top_pairwise_features_batch=top_pairwise_features[i:i+self.fine_score_batch_size]
            token_emb_batch=token_emb[i:i+self.fine_score_batch_size]
            top_indices_batch=top_indices[i:i+self.fine_score_batch_size]
            top_rough_scores_batch=top_rough_scores[i:i+self.fine_score_batch_size]
            top_corefer_scores_batch=self.get_corefer_score_batch(token_emb, token_emb_batch, top_pairwise_features_batch, top_rough_scores_batch, top_indices_batch)
            corefer_scores_list.append(top_corefer_scores_batch)
        top_corefer_scores=torch.cat(corefer_scores_list, dim=0) #top_corefer_scores(num_tokens, k+1)
        return top_corefer_scores
    
    def get_coref_y(self, top_rough_scores, top_indices, cluster_ids):
        valid_pair_map=top_rough_scores>float("-inf") #valid_pair_map(num_tokens,k), 1 for valid token-antecedent pair, 0 otherwise
        top_cluster_ids=cluster_ids[top_indices]*valid_pair_map 
        top_cluster_ids[top_cluster_ids==0]=-1
        top_cluster_ids=add_dummy(top_cluster_ids)
        top_gold_map=(top_cluster_ids==cluster_ids.unsqueeze(1)) 
        top_gold_map[top_gold_map.sum(dim=1)==0, 0]=1 #top_gold_map(num_tokens,k+1), 1 for gold pair, 0 otherwise. The token i has empty antecedent if top_gold_map[i,0]=1 
        return top_gold_map
    
    def get_span_prediction_scores(self, token_emb, heads_ids, sent_ids, training):
        span_scores=self.span_predictor(token_emb, heads_ids, sent_ids, training) #(num_heads, num_tokens, 2)
        return span_scores
    
    def get_loss(self, top_corefer_scores, coref_y, span_scores, span_y, num_gold_spans):
        coref_y=coref_y.to(torch.float)
        c_loss=self.coref_loss(top_corefer_scores, coref_y)
        if span_scores is not None:
            assert num_gold_spans>0
            s_loss=(self.span_loss(span_scores[:,:,0], span_y[0])+self.span_loss(span_scores[:,:,1], span_y[1]))/num_gold_spans/2
        else:
            s_loss=torch.zeros_like(c_loss)
        return c_loss+s_loss
    
    def forward(self, input_ids, input_mask, word2subword, speaker_ids, genre_id, sent_ids, gold_starts=None, gold_ends=None, 
                gold_mention_cluster_ids=None, cluster_ids=None, gold_heads_ids=None, training=True):
        """input_ids(num_segments,max_seg_len), input_mask(num_segments,max_seg_len)
           word2subword(num_tokens)：a list of tuples indicating which subtokens each token is mapping to (except <cls> and <sep> tokens)
           speaker_ids(num_tokens)：the speaker id of each token in the doc (except <cls> and <sep> tokens)
           genre_id(int): genre id of the doc
           sent_ids(num_tokens): the sentence id for each token (except <cls> and <sep> tokens)
           gold_starts(num_gold_spans): start token index of every gold span, gold_ends(num_gold_spans): end token index of every gold span (except <cls> and <sep> tokens)
           gold_mention_cluster_ids(num_gold_spans): cluster id of every gold span
           cluster_ids(num_tokens)： cluster id of each token in the doc, non-coreferent token has cluster id 0 (except <cls> and <sep> tokens)
           gold_heads_ids(num_gold_spans): token index of the head of each gold span
           training(bool): True for training, False for inference and evaluation
        """
        self.device=input_ids.device
        device=self.device
        
        #get the subtoken embeddings of the input
        subtoken_emb=self.get_subtoken_emb(input_ids, input_mask)
        
        #get token embeddings from subtoken embeddings
        token_emb, num_tokens, num_subtokens, token_attn_weights=self.get_token_emb(subtoken_emb, word2subword)
        
        #get rough corefer score between each token and its antecedents, keep top k antecedents for each token
        top_rough_scores, top_indices=self.get_rough_score(token_emb, num_tokens)
        
        #get pairwise features between each token and its top k antecedents
        top_pairwise_features=self.get_pairwise_features(top_indices, speaker_ids, genre_id, num_tokens)
        
        #split doc into batches, and get fine corefer scores between each token and its top k antecedents for each batch
        top_corefer_scores=self.get_corefer_score(token_emb, top_indices, top_rough_scores, top_pairwise_features, num_tokens)
        
        #get true labels for token coreference
        coref_y=self.get_coref_y(top_rough_scores, top_indices, cluster_ids)
        
        #get results for span prediction of each gold head
        num_gold_spans=gold_starts.shape[0]
        span_scores, span_y=None, None
        if(num_gold_spans>0):
            span_scores=self.get_span_prediction_scores(token_emb, gold_heads_ids, sent_ids, training) #recover spans for gold heads when computing the loss
            span_y=(gold_starts, gold_ends)
        
        #get loss
        loss=self.get_loss(top_corefer_scores, coref_y, span_scores, span_y, num_gold_spans)
        
        return loss, [token_emb, sent_ids, top_corefer_scores, top_indices, gold_starts, gold_ends, gold_mention_cluster_ids, span_scores, span_y]
    
    def get_predicted_token_clusters(self, top_corefer_scores, top_indices):
        predicted_antecedents=top_corefer_scores.argmax(dim=1)-1
        not_dummy=predicted_antecedents>=0
        #assume n is the number of tokens predicted to have non-dummy antecedent
        source_tokens=torch.arange(0, len(top_corefer_scores))[not_dummy] #source_tokens(n)
        predicted_antecedents=top_indices[source_tokens, predicted_antecedents[not_dummy]] #predicted_antecedents(n)
        n=len(source_tokens)
        
        token_to_cluster_id={} #record the cluster id for each non-dummy token
        predicted_token_clusters=[] #a list of predicted token clusters, each cluster is a list of token indices
        for i in range(n):
            source_token, antecedent_token=source_tokens[i].item(), predicted_antecedents[i].item()
            assert antecedent_token<source_token
            antecedent_cluster_id=token_to_cluster_id.get(antecedent_token, -1)
            if antecedent_cluster_id==-1:
                antecedent_cluster_id=len(predicted_token_clusters)
                predicted_token_clusters.append([antecedent_token])
                token_to_cluster_id[antecedent_token]=antecedent_cluster_id
            predicted_token_clusters[antecedent_cluster_id].append(source_token)
            token_to_cluster_id[source_token]=antecedent_cluster_id
        return predicted_token_clusters, token_to_cluster_id
        
    def get_predicted_span_clusters(self, token_emb, sent_ids, top_corefer_scores, top_indices):
        #recover spans for predicted tokens when doing the inference
        predicted_token_clusters, token_to_cluster_id=self.get_predicted_token_clusters(top_corefer_scores, top_indices)
        heads_ids=torch.tensor(sorted(i for cluster in predicted_token_clusters for i in cluster), device=self.device)
        if not predicted_token_clusters:
            return [],{}
        scores=self.span_predictor(token_emb, heads_ids, sent_ids, training=False)
        starts=(scores[:,:,0].argmax(dim=1)).tolist()
        ends=(scores[:,:,1].argmax(dim=1)).tolist()
        head2span={}
        for head, start, end in zip(heads_ids.tolist(), starts, ends):
            assert start<=end
            head2span[head]=(start,end)
        predicted_span_clusters=[[head2span[head] for head in cluster] for cluster in predicted_token_clusters] #a list of clusters of predicted spans
        predicted_span_clusters=[tuple(c) for c in predicted_span_clusters]
        mention_to_cluster_id={head2span[head]:cluster_id for head, cluster_id in token_to_cluster_id.items()} #record the cluster id for each mention
        return predicted_span_clusters, mention_to_cluster_id
        
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
        
    def get_evaluation_results(self, token_emb: torch.Tensor, sent_ids: torch.Tensor, top_corefer_scores: torch.Tensor, top_indices: torch.Tensor, gold_starts: list, gold_ends: list, gold_mention_cluster_ids: list):
        predicted_clusters, predicted_mention_to_cluster_id=self.get_predicted_span_clusters(token_emb, sent_ids, top_corefer_scores, top_indices)
        mention_to_predicted={m: predicted_clusters[cluster_idx] for m, cluster_idx in predicted_mention_to_cluster_id.items()} #dict mapping each mention to its predicted cluster of mentions
        gold_clusters, _=self.get_gold_clusters(gold_starts, gold_ends, gold_mention_cluster_ids)
        mention_to_gold={m: cluster for cluster in gold_clusters for m in cluster} #dict mapping each mention to its gold cluster of mentions
        return predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold
    
    def get_span_accuracy(self, span_scores, span_y):
        if span_scores is not None:
            pred_starts=span_scores[:, :, 0].argmax(dim=1)
            pred_ends=span_scores[:, :, 1].argmax(dim=1)
            s_correct=((span_y[0]==pred_starts)*(span_y[1]==pred_ends)).sum().item()
            return s_correct/len(pred_starts)
        else:
            return None
