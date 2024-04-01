"""Transformer lightning module based on lightning transformers."""
from typing import Any, Dict, Optional, Type

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel
from lightning_modules.models.word_level_coref_model import BertWL_Model
import sys
sys.path.append("src")
from metrics import f1, b_cubed, muc, ceafe


class WordLevelCorefModule(pl.LightningModule):
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
        model_cfg = {"bert_pretrained_name":"bert-base-uncased", "dropout_rate":0.3, "max_num_candidate_antecedents":50, "feature_emb_size":20, "num_genres":7, 
                   "fine_score_batch_size":512, "ffnn_depth":1, "ffnn_size":1024, "span_predictor_emb_size":64, "bce_loss_weight":0.5, "max_seg_len":512}
        self.model = BertWL_Model(model_cfg, self.device)

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-05) #To change
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
        token_emb, sent_ids, top_corefer_scores, top_indices, gold_starts, gold_ends, gold_mention_cluster_ids=output_list[0], output_list[1], output_list[2], output_list[3], output_list[4], output_list[5], output_list[6] 
        gold_starts, gold_ends, gold_mention_cluster_ids=gold_starts.tolist(), gold_ends.tolist(), gold_mention_cluster_ids.tolist()
        predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold=self.model.get_evaluation_results(token_emb, sent_ids, top_corefer_scores, top_indices, gold_starts, gold_ends, gold_mention_cluster_ids)
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
        
        span_scores, span_y=output_list[7], output_list[8]
        span_accuracy=self.model.get_span_accuracy(span_scores, span_y)
        return {"ceafe": [recall_ceafe, precision_ceafe, f1_score_ceafe], "b3": [recall_b3, precision_b3, f1_score_b3], "muc": [recall_muc, precision_muc, f1_score_muc], "avg": [recall_avg, precision_avg, f1_score_avg], "sa": span_accuracy}

    def training_step(self, batch, batch_idx):
        loss, output_list = self._step(batch, batch_idx)
        token_emb, sent_ids, top_corefer_scores, top_indices, gold_starts, gold_ends, gold_mention_cluster_ids=output_list[0], output_list[1], output_list[2], output_list[3], output_list[4], output_list[5], output_list[6] 
        gold_starts, gold_ends, gold_mention_cluster_ids=gold_starts.tolist(), gold_ends.tolist(), gold_mention_cluster_ids.tolist()
        predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold=self.model.get_evaluation_results(token_emb, sent_ids, top_corefer_scores, top_indices, gold_starts, gold_ends, gold_mention_cluster_ids)
        print(len(mention_to_predicted))
        print(len(mention_to_gold))
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
        if eval_dict["sa"] is not None:
            self.log('span accuracy', eval_dict["sa"], sync_dist=True, prog_bar=True)

        
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
        
