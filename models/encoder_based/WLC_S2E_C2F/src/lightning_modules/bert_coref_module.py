"""Transformer lightning module based on lightning transformers."""
from typing import Any, Dict, Optional, Type

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel
from lightning_modules.models.bert_coref_model import BertCorefModel
import sys
sys.path.append("src")
from metrics import f1, b_cubed, muc, ceafe


class BertCorefModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 2e-5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.initialize_model()

    def initialize_model(self):
        # TODO: move to config
        model_cfg = {"feature_emb_size":20, "dropout_rate":0.3, "bert_pretrained_name":"bert-base-uncased", "max_span_width":30, 
            "max_segment_len":128, "use_span_width_to_compute_emb":True, "span_emb_compute_method":"attention", "ffnn_size":3000, "ffnn_depth":1,
            "max_num_extracted_spans":3900, "top_span_ratio":0.4, "crossing_mentions_allowed":False, "max_top_antecedents": 50, 
            "use_span_dist_to_compute_rough_score":True, "use_fine_score":True, "use_speaker_info_to_compute_fine_score":True,
            "use_genre_info_to_compute_fine_score":True, "use_seg_dist_to_compute_fine_score":True, "use_antecedent_dist_to_compute_fine_score":True, 
            "use_span_width_to_compute_mention_score": True, "num_genres":7, "max_num_segments":11}
        self.model = BertCorefModel(model_cfg, self.device)

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-05,  eps = 1e-6, weight_decay = 1e-2) #To change
        return {
            'optimizer': optimizer,
        }

    def _step(self, batch, batch_idx):
        # make sure batch size is one and squeeze
        for k, v in batch.items():
            assert v.shape[0] == 1
            batch[k] = v.squeeze(0)
        
        outputs = self.model(**batch)
        return outputs
    
    def _eval(self, output_list):
        # evaluate the model output, return recall, precision and f1 score under each of ceafe, b3 and muc metrics
        top_span_starts, top_span_ends, top_antecedent_ids, top_antecedent_scores, gold_starts, gold_ends, gold_mention_cluster_ids=output_list[3], output_list[4], output_list[5], output_list[6], output_list[7], output_list[8], output_list[9] 
        top_span_starts, top_span_ends, top_antecedent_ids, top_antecedent_scores, gold_starts, gold_ends, gold_mention_cluster_ids=top_span_starts.tolist(), top_span_ends.tolist(), top_antecedent_ids.tolist(), top_antecedent_scores.tolist(), gold_starts.tolist(), gold_ends.tolist(), gold_mention_cluster_ids.tolist()
        predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold=self.model.get_evaluation_results(top_span_starts, top_span_ends, top_antecedent_ids, top_antecedent_scores, gold_starts, gold_ends, gold_mention_cluster_ids)
        pn, pd, rn, rd=ceafe(predicted_clusters, gold_clusters)
        recall_ceafe = 0 if rn == 0 else rn / float(rd)
        precision_ceafe = 0 if pn == 0 else pn / float(pd)
        f1_score_ceafe = f1(pn, pd, rn, rd, beta=1)
        pn, pd = b_cubed(predicted_clusters, mention_to_gold)
        rn, rd = b_cubed(gold_clusters, mention_to_predicted)
        recall_b3 = 0 if rn == 0 else rn / float(rd)
        precision_b3 = 0 if pn == 0 else pn / float(pd)
        f1_score_b3 = f1(pn, pd, rn, rd, beta=1)
        pn, pd = muc(predicted_clusters, mention_to_gold)
        rn, rd = muc(gold_clusters, mention_to_predicted)
        recall_muc = 0 if rn == 0 else rn / float(rd)
        precision_muc = 0 if pn == 0 else pn / float(pd)
        f1_score_muc = f1(pn, pd, rn, rd, beta=1)
        recall_avg, precision_avg, f1_score_avg = (recall_ceafe+recall_b3+recall_muc)/3, (precision_ceafe+precision_b3+precision_muc)/3, (f1_score_ceafe+f1_score_b3+f1_score_muc)/3
        return {"ceafe": [recall_ceafe, precision_ceafe, f1_score_ceafe], "b3": [recall_b3, precision_b3, f1_score_b3], "muc": [recall_muc, precision_muc, f1_score_muc], "avg": [recall_avg, precision_avg, f1_score_avg]}

    def training_step(self, batch, batch_idx):
        loss, _ = self._step(batch, batch_idx)
        self.log('train loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, output_list = self._step(batch, batch_idx)
        eval_dict=self._eval(output_list)
        self.log('validation loss', loss, sync_dist=True, prog_bar=True)
        self.log('ceafe recall', eval_dict["ceafe"][0], sync_dist=True, prog_bar=True)
        self.log('ceafe precision', eval_dict["ceafe"][1], sync_dist=True, prog_bar=True)
        self.log('ceafe f1', eval_dict["ceafe"][2], sync_dist=True, prog_bar=True)
        self.log('b3 recall', eval_dict["b3"][0], sync_dist=True, prog_bar=True)
        self.log('b3 precision', eval_dict["b3"][1], sync_dist=True, prog_bar=True)
        self.log('b3 f1', eval_dict["b3"][2], sync_dist=True, prog_bar=True)
        self.log('muc recall', eval_dict["muc"][0], sync_dist=True, prog_bar=True)
        self.log('muc precision', eval_dict["muc"][1], sync_dist=True, prog_bar=True)
        self.log('muc f1', eval_dict["muc"][2], sync_dist=True, prog_bar=True)
        self.log('average recall', eval_dict["avg"][0], sync_dist=True, prog_bar=True)
        self.log('average precision', eval_dict["avg"][1], sync_dist=True, prog_bar=True)
        self.log('average f1', eval_dict["avg"][2], sync_dist=True, prog_bar=True)
        
        
    def test_step(self, batch, batch_idx):
        loss, output_list = self._step(batch, batch_idx)
        eval_dict=self._eval(output_list)
        self.log('test loss', loss, sync_dist=True, prog_bar=True)
        self.log('ceafe recall', eval_dict["ceafe"][0], sync_dist=True, prog_bar=True)
        self.log('ceafe precision', eval_dict["ceafe"][1], sync_dist=True, prog_bar=True)
        self.log('ceafe f1', eval_dict["ceafe"][2], sync_dist=True, prog_bar=True)
        self.log('b3 recall', eval_dict["b3"][0], sync_dist=True, prog_bar=True)
        self.log('b3 precision', eval_dict["b3"][1], sync_dist=True, prog_bar=True)
        self.log('b3 f1', eval_dict["b3"][2], sync_dist=True, prog_bar=True)
        self.log('muc recall', eval_dict["muc"][0], sync_dist=True, prog_bar=True)
        self.log('muc precision', eval_dict["muc"][1], sync_dist=True, prog_bar=True)
        self.log('muc f1', eval_dict["muc"][2], sync_dist=True, prog_bar=True)
        self.log('average recall', eval_dict["avg"][0], sync_dist=True, prog_bar=True)
        self.log('average precision', eval_dict["avg"][1], sync_dist=True, prog_bar=True)
        self.log('average f1', eval_dict["avg"][2], sync_dist=True, prog_bar=True)
        
