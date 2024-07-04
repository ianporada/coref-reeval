import sys
sys.path.append('/home/mila/x/xiyuan.zou/research/kd-coref')
from typing import Any, Dict, Optional, Type
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
import torch
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from datasets import Dataset, load_dataset
from evaluators.metrics import muc, b_cubed, ceafe, f1
from evaluators.evaluators import CorefEvaluator, MentionEvaluator
from models.s2e_model import S2E_model
from models.lingmess_model import LingMess_model


class TrainingModule_distill_from_lingmess_to_s2e(pl.LightningModule):
    def __init__(
        self,
        preprocessing_cfg_teacher, 
        model_cfg_teacher, 
        training_cfg_teacher, 
        preprocessing_cfg_student,
        model_cfg_student,
        training_cfg_student
    ) -> None:
        super().__init__()
        self.preprocessing_cfg_teacher=preprocessing_cfg_teacher
        self.model_cfg_teacher=model_cfg_teacher
        self.training_cfg_teacher=training_cfg_teacher
        self.preprocessing_cfg_student=preprocessing_cfg_student
        self.model_cfg_student=model_cfg_student
        self.training_cfg_student=training_cfg_student
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
        dataset_train=S2E_Dataset(dataset_raw["train"], tokenizer, self.preprocessing_cfg["max_seq_len"])
        dataset_val=S2E_Dataset(dataset_raw["validation"], tokenizer, self.preprocessing_cfg["max_seq_len"])
        dataset_test=S2E_Dataset(dataset_raw["test"], tokenizer, self.preprocessing_cfg["max_seq_len"])    
        return dataset_train, dataset_val, dataset_test
    
    def prepare_data(self) -> None:
        dataset_raw=load_dataset('conll2012_ontonotesv5', 'english_v4', cache_dir=self.preprocessing_cfg["cache_dir"])
        dataset_raw=self.get_doc_parts(dataset_raw) # split each document into document part since the annotation is made with regard of each document part
        self.dataset_train, self.dataset_val, self.dataset_test=self.get_transformed_dataset(dataset_raw)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return S2E_Dataloader(self.dataset_train, num_workers=self.preprocessing_cfg["num_workers"], max_total_seq_len=self.preprocessing_cfg["max_total_seq_len"], is_training=True)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return S2E_Dataloader(self.dataset_val, num_workers=self.preprocessing_cfg["num_workers"], max_total_seq_len=self.preprocessing_cfg["max_total_seq_len"], is_training=False)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return S2E_Dataloader(self.dataset_test, num_workers=self.preprocessing_cfg["num_workers"], max_total_seq_len=self.preprocessing_cfg["max_total_seq_len"], is_training=False)
    
    #preparing training    
    def initialize_model(self):
        self.model_student=S2E_model(self.model_cfg_student)
        self.model_teacher=LingMess_model(self.model_cfg_teacher)
        
    def initialize_evaluators(self):
        self.post_pruning_mention_evaluator = MentionEvaluator()
        self.mention_evaluator = MentionEvaluator()
        self.coref_evaluator = CorefEvaluator()

    def configure_optimizers(self) -> Dict:
        no_decay = ['bias', 'LayerNorm.weight']
        head_params = ['coref', 'mention', 'antecedent']
        model_decay = [p for n, p in self.model_student.named_parameters() if not any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay)]
        model_no_decay = [p for n, p in self.model_student.named_parameters() if not any(hp in n for hp in head_params) and any(nd in n for nd in no_decay)]
        head_decay = [p for n, p in self.model_student.named_parameters() if any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay)]
        head_no_decay = [p for n, p in self.model_student.named_parameters() if any(hp in n for hp in head_params) and any(nd in n for nd in no_decay)]
        
        optimizer_grouped_parameters = [
            {'params': model_decay, 'lr': self.training_cfg_student["encoder_lr"], 'weight_decay': self.training_cfg_student["weight_decay"]},
            {'params': model_no_decay, 'lr': self.training_cfg_student["encoder_lr"], 'weight_decay': 0.0},
            {'params': head_decay, 'lr': self.training_cfg_student["head_lr"], 'weight_decay': self.training_cfg_student["weight_decay"]},
            {'params': head_no_decay, 'lr': self.training_cfg_student["head_lr"], 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.training_cfg_student["encoder_lr"], betas=(self.training_cfg_student["adam_beta1"], self.training_cfg_student["adam_beta2"]), eps=self.training_cfg_student["adam_eps"])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.training_cfg_student["num_warmup_steps"], num_training_steps=self.training_cfg_student["num_epochs"]*self.training_cfg_student["total_steps_per_epoch"])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        }
    
    