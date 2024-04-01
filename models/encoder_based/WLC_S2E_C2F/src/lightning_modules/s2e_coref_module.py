"""Transformer lightning module based on lightning transformers."""
from typing import Any, Dict, Optional, Type

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from transformers import LongformerModel
from lightning_modules.models.s2e_coref_model import Longformer_S2E_model
import sys
sys.path.append("src")
from metrics import f1, b_cubed, muc, ceafe


class S2ECorefModule(pl.LightningModule):
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
        model_cfg = {"longformer_pretrained_name":"allenai/longformer-large-4096", "max_span_len":30, "top_lambda":0.4, "ffnn_size":3072, "dropout_prob":0.3, "normalise_loss":True}
        self.model = Longformer_S2E_model(model_cfg, self.device)

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-05,  eps = 1e-6, weight_decay = 1e-2) #To change
        return {
            'optimizer': optimizer,
        }

    def _step(self, batch, batch_idx):
        outputs = self.model(batch[0], batch[1], batch[2])
        return outputs
    
    def _eval(self, output_list):
        # evaluate the model output, return recall, precision and f1 score under each of ceafe, b3 and muc metrics
        start_ids, end_ids, final_scores, gold_clusters=output_list[0], output_list[1], output_list[2], output_list[3]
        start_ids, end_ids, final_scores, gold_clusters=start_ids.squeeze(0).cpu().numpy(), end_ids.squeeze(0).cpu().numpy(), final_scores.squeeze(0).cpu().numpy(), gold_clusters.squeeze(0).cpu().numpy()
        predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold=self.model.get_evaluation_results(start_ids, end_ids, final_scores, gold_clusters)
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
        assert len(batch[0])==1
        assert len(batch[1])==1
        assert len(batch[2])==1
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
        assert len(batch[0])==1
        assert len(batch[1])==1
        assert len(batch[2])==1
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