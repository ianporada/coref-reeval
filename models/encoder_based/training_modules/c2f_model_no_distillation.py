import sys
sys.path.append('/home/mila/x/xiyuan.zou/research/kd-coref')
from typing import Any, Dict, Optional, Type
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
import torch
from torch import nn
from models.c2f_model import C2F_model
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from datasets import Dataset, load_dataset
from data_preprocessing.c2f_data_processor import C2F_Dataset, C2F_Dataloader
from evaluators.metrics import muc, b_cubed, ceafe, f1
from evaluators.evaluators import CorefEvaluator, MentionEvaluator
from torch.optim.lr_scheduler import LambdaLR


class TrainingModule_c2f_no_distillation(pl.LightningModule):
    def __init__(
        self,
        preprocessing_cfg, 
        model_cfg, 
        training_cfg, 
    ) -> None:
        super().__init__()
        self.preprocessing_cfg=preprocessing_cfg
        self.model_cfg=model_cfg
        self.training_cfg=training_cfg
        self.save_hyperparameters()
        self.initialize_model()
        self.initialize_evaluators()
    
    #data preprocessing 
    def get_doc_parts(self, dataset: Dataset):
        return dataset.map(
            self.split_doc_into_docParts,
            batched=True,
            batch_size=1
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
    
    def get_transformed_dataset(self, dataset_raw: Dataset):
        tokenizer=AutoTokenizer.from_pretrained(self.preprocessing_cfg["tokenizer_name"])
        dataset_train=C2F_Dataset(dataset_raw["train"], tokenizer, is_training=True, config=self.preprocessing_cfg)
        dataset_val=C2F_Dataset(dataset_raw["validation"], tokenizer, is_training=False, config=self.preprocessing_cfg)
        dataset_test=C2F_Dataset(dataset_raw["test"], tokenizer, is_training=False, config=self.preprocessing_cfg)    
        return dataset_train, dataset_val, dataset_test
    
    def prepare_data(self) -> None:
        dataset_raw=load_dataset('conll2012_ontonotesv5', 'english_v4', cache_dir=self.preprocessing_cfg["cache_dir"])
        dataset_raw=self.get_doc_parts(dataset_raw) # split each document into document part since the annotation is made with regard of each document part
        self.dataset_train, self.dataset_val, self.dataset_test=self.get_transformed_dataset(dataset_raw)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return C2F_Dataloader(self.dataset_train, num_workers=self.preprocessing_cfg["num_workers"])
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return C2F_Dataloader(self.dataset_val, num_workers=self.preprocessing_cfg["num_workers"])
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return C2F_Dataloader(self.dataset_test, num_workers=self.preprocessing_cfg["num_workers"])
    
    #preparing training    
    def initialize_model(self):
        self.model=C2F_model(self.model_cfg)
        
    def initialize_evaluators(self):
        self.post_pruning_mention_evaluator = MentionEvaluator()
        self.mention_evaluator = MentionEvaluator()
        self.coref_evaluator = CorefEvaluator()

    def get_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        head_params = ['span', 'mention', 'antecedent', 'coarse_bilinear', 'speaker', 'genre', 'distance', 'ffnn']
        model_decay = [p for n, p in self.model.named_parameters() if not any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay)]
        model_no_decay = [p for n, p in self.model.named_parameters() if not any(hp in n for hp in head_params) and any(nd in n for nd in no_decay)]
        head_decay = [p for n, p in self.model.named_parameters() if any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay)]
        head_no_decay = [p for n, p in self.model.named_parameters() if any(hp in n for hp in head_params) and any(nd in n for nd in no_decay)]
        
        for n, p in self.model.named_parameters():
            if any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay):
                print(n)
            if any(hp in n for hp in head_params) and any(nd in n for nd in no_decay):
                print(n)
        
        optimizer_grouped_parameters = [
            {'params': model_decay, 'lr': self.training_cfg["encoder_lr"], 'weight_decay': self.training_cfg["weight_decay"]},
            {'params': model_no_decay, 'lr': self.training_cfg["encoder_lr"], 'weight_decay': 0.0},
            {'params': head_decay, 'lr': self.training_cfg["head_lr"], 'weight_decay': self.training_cfg["weight_decay"]},
            {'params': head_no_decay, 'lr': self.training_cfg["head_lr"], 'weight_decay': 0.0}
        ]
        optimizer=AdamW(optimizer_grouped_parameters, lr=self.training_cfg["encoder_lr"], eps=self.training_cfg["adam_eps"])
        return optimizer

    def get_scheduler(self, optimizer, total_update_steps):
        # Only warm up bert lr
        warmup_steps = int(total_update_steps * 0.1) #warm_up ratio fixed at 0.1

        def lr_lambda_bert(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps - warmup_steps))
            )

        def lr_lambda_task(current_step):
            return max(0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps)))

        scheduler=LambdaLR(optimizer, [lr_lambda_bert, lr_lambda_bert, lr_lambda_task, lr_lambda_task])
        return scheduler

    def configure_optimizers(self) -> Dict:
        total_update_steps=self.training_cfg["num_epochs"]*self.training_cfg["total_steps_per_epoch"]
        optimizer=self.get_optimizer()
        scheduler=self.get_scheduler(optimizer, total_update_steps)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        }
        
    #training logics
    def _step(self, batch, batch_idx):
        # make sure batch size is one and squeeze
        outputs = self.model(*batch)
        return outputs
    
    def _eval(self, output_list):
        # evaluate the model output and update the evaluators
        top_span_starts, top_span_ends, top_antecedent_ids, top_antecedent_scores, gold_starts, gold_ends, gold_mention_cluster_ids=output_list[3], output_list[4], output_list[5], output_list[6], output_list[7], output_list[8], output_list[9] 
        top_span_starts, top_span_ends, top_antecedent_ids, top_antecedent_scores, gold_starts, gold_ends, gold_mention_cluster_ids=top_span_starts.tolist(), top_span_ends.tolist(), top_antecedent_ids.tolist(), top_antecedent_scores.tolist(), gold_starts.tolist(), gold_ends.tolist(), gold_mention_cluster_ids.tolist()
        predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold=self.model.get_evaluation_results(top_span_starts, top_span_ends, top_antecedent_ids, top_antecedent_scores, gold_starts, gold_ends, gold_mention_cluster_ids)
        predicted_mentions=list(mention_to_predicted.keys())
        gold_mentions=list(mention_to_gold.keys())
        candidate_mentions=list(zip(top_span_starts, top_span_ends))
        self.coref_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        self.mention_evaluator.update(predicted_mentions, gold_mentions)
        self.post_pruning_mention_evaluator.update(candidate_mentions, gold_mentions)
        
    def training_step(self, batch, batch_idx):
        loss, _ = self._step(batch, batch_idx)
        self.log('train loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, output_list = self._step(batch, batch_idx)
        self._eval(output_list)
        self.log('validation loss', loss)
    
    def on_validation_epoch_end(self):
        post_pruning_mention_precision, post_pruning_mentions_recall, post_pruning_mention_f1=self.post_pruning_mention_evaluator.get_prf()
        mention_precision, mention_recall, mention_f1=self.mention_evaluator.get_prf()
        coref_precision, coref_recall, coref_f1=self.coref_evaluator.get_prf()
        muc_precision, muc_recall, muc_f1=self.coref_evaluator.get_muc()
        b3_precision, b3_recall, b3_f1=self.coref_evaluator.get_b3()
        ceafe_precision, ceafe_recall, ceafe_f1=self.coref_evaluator.get_ceafe()
        self.post_pruning_mention_evaluator.clear() #clear all the evaluators
        self.mention_evaluator.clear()
        self.coref_evaluator.clear()
        self.log("post pruning mention precision", post_pruning_mention_precision)
        self.log("post pruning mention recall", post_pruning_mentions_recall)
        self.log("post pruning mention f1", post_pruning_mention_f1)
        self.log("mention precision", mention_precision)
        self.log("mention recall", mention_recall)
        self.log("mention f1", mention_f1)
        self.log("coref avg precision", coref_precision)
        self.log("coref avg recall", coref_recall)
        self.log("coref avg f1", coref_f1)
        self.log("coref muc precision", muc_precision)
        self.log("coref muc recall", muc_recall)
        self.log("coref muc f1", muc_f1)
        self.log("coref b3 precision", b3_precision)
        self.log("coref b3 recall", b3_recall)
        self.log("coref b3 f1", b3_f1)
        self.log("coref ceafe precision", ceafe_precision)
        self.log("coref ceafe recall", ceafe_recall)
        self.log("coref ceafe f1", ceafe_f1)
        
    def test_step(self, batch, batch_idx):
        loss, output_list = self._step(batch, batch_idx)
        self._eval(output_list)
        
    def on_test_epoch_end(self):
        post_pruning_mention_precision, post_pruning_mentions_recall, post_pruning_mention_f1=self.post_pruning_mention_evaluator.get_prf()
        mention_precision, mention_recall, mention_f1=self.mention_evaluator.get_prf()
        coref_precision, coref_recall, coref_f1=self.coref_evaluator.get_prf()
        muc_precision, muc_recall, muc_f1=self.coref_evaluator.get_muc()
        b3_precision, b3_recall, b3_f1=self.coref_evaluator.get_b3()
        ceafe_precision, ceafe_recall, ceafe_f1=self.coref_evaluator.get_ceafe()
        self.post_pruning_mention_evaluator.clear() #clear all the evaluators
        self.mention_evaluator.clear()
        self.coref_evaluator.clear()
        self.log("post pruning mention precision", post_pruning_mention_precision)
        self.log("post pruning mention recall", post_pruning_mentions_recall)
        self.log("post pruning mention f1", post_pruning_mention_f1)
        self.log("mention precision", mention_precision)
        self.log("mention recall", mention_recall)
        self.log("mention f1", mention_f1)
        self.log("coref precision", coref_precision)
        self.log("coref recall", coref_recall)
        self.log("coref f1", coref_f1)
        self.log("coref muc precision", muc_precision)
        self.log("coref muc recall", muc_recall)
        self.log("coref muc f1", muc_f1)
        self.log("coref b3 precision", b3_precision)
        self.log("coref b3 recall", b3_recall)
        self.log("coref b3 f1", b3_f1)
        self.log("coref ceafe precision", ceafe_precision)
        self.log("coref ceafe recall", ceafe_recall)
        self.log("coref ceafe f1", ceafe_f1)
        