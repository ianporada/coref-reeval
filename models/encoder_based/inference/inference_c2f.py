import sys
sys.path.append('/home/mila/x/xiyuan.zou/research/kd-coref')
import hydra
import numpy as np
import torch
import csv
from tqdm import tqdm
from omegaconf import DictConfig
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from training_modules.c2f_model_no_distillation import TrainingModule_c2f_no_distillation
from data_preprocessing.c2f_data_processor import C2F_Dataset, C2F_Dataloader, C2F_Dataset_GAP, C2F_Dataset_ontoGUM, C2F_Dataset_official
from evaluators.evaluators import CorefEvaluator, MentionEvaluator
from evaluators.metrics import total_num_parameters, get_peak_memory
from evaluators.gap_scorer import run_scorer
from utilities.convert_to_conll import write_docs_in_conll_format
import os
import re
import subprocess
from utilities.utils import extract_r_p_f


def get_doc_parts_official(dataset: Dataset): #for official evaluation on ontonotes
        return dataset.map(
            split_doc_into_docParts_official,
            batched=True,
            batch_size=1
        )  
        
def split_doc_into_docParts_official(example): #for official evaluation on ontonotes
    """take a doc and return the doc parts"""
    docParts_dict={} #{part0:[], part1:[]...}
    for sent_dict in example["sentences"][0]:
        sent_part_id=sent_dict["part_id"]
        if sent_part_id in docParts_dict:
            docParts_dict[sent_part_id].append(sent_dict)
        else:
            docParts_dict[sent_part_id]=[sent_dict]
    document_id=example["document_id"][0]
    num_parts=len(docParts_dict)
    return {"document_id": [f'{document_id}/part_{k}' for k in docParts_dict], "sentences": [docParts_dict[k] for k in docParts_dict]}


@hydra.main(config_path='../config', config_name='config_inference_c2f')
def main(cfg : DictConfig) -> None:
    #load trained model
    training_module=TrainingModule_c2f_no_distillation.load_from_checkpoint(cfg["model_ckpt"])
    model=training_module.model
    if cfg["inference_device"]=="gpu":
        model.cuda(0)
    model.eval()
    
    #prepare inference datasets
    if cfg["dataset_name"]=="ontonotes":
        if cfg["use_official_scorer"]==False:
            dataset_raw=load_dataset('conll2012_ontonotesv5', 'english_v4', cache_dir=cfg["cache_dir"])
            dataset_raw=training_module.get_doc_parts(dataset_raw)
            tokenizer=AutoTokenizer.from_pretrained(cfg["preprocessing_cfg"]["tokenizer_name"])
            if cfg["dataset_split"]=="train":
                dataset=C2F_Dataset(dataset_raw["train"], tokenizer, is_training=False, config=cfg["preprocessing_cfg"])
            elif cfg["dataset_split"]=="val":
                dataset=C2F_Dataset(dataset_raw["validation"], tokenizer, is_training=False, config=cfg["preprocessing_cfg"])
            elif cfg["dataset_split"]=="test":
                dataset=C2F_Dataset(dataset_raw["test"], tokenizer, is_training=False, config=cfg["preprocessing_cfg"])
            dataloader=C2F_Dataloader(dataset, num_workers=1)
        else:
            dataset_raw=load_dataset('conll2012_ontonotesv5', 'english_v4', cache_dir=cfg["cache_dir"])
            dataset_raw=get_doc_parts_official(dataset_raw)
            tokenizer=AutoTokenizer.from_pretrained(cfg["preprocessing_cfg"]["tokenizer_name"])
            if cfg["dataset_split"]=="train":
                dataset=C2F_Dataset_official(dataset_raw["train"], tokenizer, config=cfg["preprocessing_cfg"])
            elif cfg["dataset_split"]=="val":
                dataset=C2F_Dataset_official(dataset_raw["validation"], tokenizer, config=cfg["preprocessing_cfg"])
            elif cfg["dataset_split"]=="test":
                dataset=C2F_Dataset_official(dataset_raw["test"], tokenizer, config=cfg["preprocessing_cfg"])
            dataloader=C2F_Dataloader(dataset, num_workers=1)
    elif cfg["dataset_name"]=="GAP":
        if cfg["use_official_scorer"]==False:
            assert False, "only official scorer supported for GAP dataset"
        else:
            dataset_raw=load_dataset("gap", cache_dir=cfg["cache_dir"])
            tokenizer=AutoTokenizer.from_pretrained(cfg["preprocessing_cfg"]["tokenizer_name"])
            if cfg["dataset_split"]=="train":
                dataset=C2F_Dataset_GAP(dataset_raw["train"], tokenizer, config=cfg["preprocessing_cfg"])
            elif cfg["dataset_split"]=="val":
                dataset=C2F_Dataset_GAP(dataset_raw["validation"], tokenizer, config=cfg["preprocessing_cfg"])
            elif cfg["dataset_split"]=="test":
                dataset=C2F_Dataset_GAP(dataset_raw["test"], tokenizer, config=cfg["preprocessing_cfg"])
            dataloader=C2F_Dataloader(dataset, num_workers=1)
    elif cfg["dataset_name"]=="ontoGUM":
        if cfg["use_official_scorer"]==False:
            dataset_raw=load_dataset('llm-coref/ontogum-full-dataset', cache_dir=cfg["cache_dir"])
            tokenizer=AutoTokenizer.from_pretrained(cfg["preprocessing_cfg"]["tokenizer_name"])
            if cfg["dataset_split"]=="train":
                dataset=C2F_Dataset_ontoGUM(dataset_raw["train"], tokenizer, config=cfg["preprocessing_cfg"])
            else:
                assert False, "ontoGUM dataset only has training split"
            dataloader=C2F_Dataloader(dataset, num_workers=1)
        else:
            assert False, "only unofficial scorer supported for ontoGUM dataset"

    #evaluate model on the ontonotes or ontoGUM unofficially
    if (cfg["dataset_name"]=="ontonotes" or cfg["dataset_name"]=="ontoGUM") and cfg["use_official_scorer"]==False:
        post_pruning_mention_evaluator=MentionEvaluator()
        mention_evaluator=MentionEvaluator()
        coref_evaluator=CorefEvaluator()
        
        with torch.no_grad():
            total_inference_time_single_batch=0
            total_memory_single_batch=0
            total_inference_time_max_batch=0
            
            for batch in iter(tqdm(dataloader)):
                if cfg["inference_device"]=="gpu":
                    batch=[batch[i].cuda(0) for i in range(9)]
                
                device=batch[0].device
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                before_memory=get_peak_memory(device) #measure the average largest inference memory of a doc with batch_size=1 
                starter, ender=torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True) #measure the average inference time of a doc with batch_size=1
                starter.record()
                _, output_list=model(*batch)
                ender.record()
                torch.cuda.synchronize()
                inference_time=starter.elapsed_time(ender)
                total_inference_time_single_batch+=inference_time
                after_memory=get_peak_memory(device)
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.empty_cache()
                total_memory_single_batch+=after_memory-before_memory
                
                model_size=total_num_parameters(model)
                
                top_span_starts, top_span_ends, top_antecedent_ids, top_antecedent_scores, gold_starts, gold_ends, gold_mention_cluster_ids=output_list[3], output_list[4], output_list[5], output_list[6], output_list[7], output_list[8], output_list[9] 
                top_span_starts, top_span_ends, top_antecedent_ids, top_antecedent_scores, gold_starts, gold_ends, gold_mention_cluster_ids=top_span_starts.tolist(), top_span_ends.tolist(), top_antecedent_ids.tolist(), top_antecedent_scores.tolist(), gold_starts.tolist(), gold_ends.tolist(), gold_mention_cluster_ids.tolist()
                predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold=model.get_evaluation_results(top_span_starts, top_span_ends, top_antecedent_ids, top_antecedent_scores, gold_starts, gold_ends, gold_mention_cluster_ids)
                predicted_mentions=list(mention_to_predicted.keys())
                gold_mentions=list(mention_to_gold.keys())
                candidate_mentions=list(zip(top_span_starts, top_span_ends))
                coref_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
                mention_evaluator.update(predicted_mentions, gold_mentions)
                post_pruning_mention_evaluator.update(candidate_mentions, gold_mentions)
        
        #print out results
        post_pruning_mention_precision, post_pruning_mentions_recall, post_pruning_mention_f1=post_pruning_mention_evaluator.get_prf()
        mention_precision, mention_recall, mention_f1=mention_evaluator.get_prf()
        coref_precision, coref_recall, coref_f1=coref_evaluator.get_prf()
        muc_precision, muc_recall, muc_f1=coref_evaluator.get_muc()
        b3_precision, b3_recall, b3_f1=coref_evaluator.get_b3()
        ceafe_precision, ceafe_recall, ceafe_f1=coref_evaluator.get_ceafe()
        post_pruning_mention_evaluator.clear() #clear all the evaluators
        mention_evaluator.clear()
        coref_evaluator.clear()
        print("coref f1", coref_f1)
        print("coref precision", coref_precision)
        print("coref recall", coref_recall)
        print("coref muc f1", muc_f1)
        print("coref muc precision", muc_precision)
        print("coref muc recall", muc_recall)
        print("coref b3 f1", b3_f1)
        print("coref b3 precision", b3_precision)
        print("coref b3 recall", b3_recall)
        print("coref ceafe f1", ceafe_f1)
        print("coref ceafe precision", ceafe_precision)
        print("coref ceafe recall", ceafe_recall)
        print("mention f1", mention_f1)
        print("mention precision", mention_precision)
        print("mention recall", mention_recall)
        print("post pruning mention f1", post_pruning_mention_f1)
        print("post pruning mention precision", post_pruning_mention_precision)
        print("post pruning mention recall", post_pruning_mentions_recall)    
        
        model_size=model_size/1000000
        print("total number of model parameters in millions", model_size)
        avg_inference_time_single_batch=total_inference_time_single_batch/len(dataloader)
        print("average inference time in milliseconds per document using single batch size", avg_inference_time_single_batch)
        avg_memory_single_batch=total_memory_single_batch/len(dataloader)
        print("average peak inference memory in MB per document using single batch size", avg_memory_single_batch)
        
        
    #evaluate model on the ontonotes officially
    if cfg["dataset_name"]=="ontonotes" and cfg["use_official_scorer"]==True: 
        with torch.no_grad():
            total_inference_time_single_batch=0
            total_memory_single_batch=0
            docs_in_huggingface_format={}
            
            for batch in iter(tqdm(dataloader)):
                official_eval_info=batch[-1]
                if cfg["inference_device"]=="gpu":
                    batch=[batch[i].cuda(0) for i in range(9)]
                device=batch[0].device
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                before_memory=get_peak_memory(device) #measure the average largest inference memory of a doc with batch_size=1 
                starter, ender=torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True) #measure the average inference time of a doc with batch_size=1
                starter.record()
                _, output_list=model(*batch)
                ender.record()
                torch.cuda.synchronize()
                inference_time=starter.elapsed_time(ender)
                total_inference_time_single_batch+=inference_time
                after_memory=get_peak_memory(device)
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.empty_cache()
                total_memory_single_batch+=after_memory-before_memory
                model_size=total_num_parameters(model)
                top_span_starts, top_span_ends, top_antecedent_ids, top_antecedent_scores, gold_starts, gold_ends, gold_mention_cluster_ids=output_list[3], output_list[4], output_list[5], output_list[6], output_list[7], output_list[8], output_list[9] 
                top_span_starts, top_span_ends, top_antecedent_ids, top_antecedent_scores, gold_starts, gold_ends, gold_mention_cluster_ids=top_span_starts.tolist(), top_span_ends.tolist(), top_antecedent_ids.tolist(), top_antecedent_scores.tolist(), gold_starts.tolist(), gold_ends.tolist(), gold_mention_cluster_ids.tolist()
                predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold=model.get_evaluation_results(top_span_starts, top_span_ends, top_antecedent_ids, top_antecedent_scores, gold_starts, gold_ends, gold_mention_cluster_ids)
                
                doc_id, hf_sentences, sentence_map_list, subtoken_map, global_word_idx_to_local_word_idx=official_eval_info[0], official_eval_info[1], official_eval_info[2], official_eval_info[3], official_eval_info[4]
                cluster_id_to_spans={}
                for cluster_id, cluster in enumerate(predicted_clusters):
                    cluster_id_to_spans[cluster_id]=[]
                    for pred_span in cluster:
                        pred_span_start_subtoken_idx=pred_span[0]
                        pred_span_end_subtoken_idx=pred_span[1]
                        sentence_id=sentence_map_list[pred_span_start_subtoken_idx]
                        pred_span_start_local_word_idx=global_word_idx_to_local_word_idx[subtoken_map[pred_span_start_subtoken_idx]]
                        pred_span_end_local_word_idx=global_word_idx_to_local_word_idx[subtoken_map[pred_span_end_subtoken_idx]]
                        cluster_id_to_spans[cluster_id].append((sentence_id, pred_span_start_local_word_idx, pred_span_end_local_word_idx))
                             
                docs_in_huggingface_format[doc_id]=(hf_sentences, cluster_id_to_spans)
            
            write_docs_in_conll_format(docs_in_huggingface_format, "/network/scratch/x/xiyuan.zou/kd-coref-project-output/results_file.conll")
            model_size=model_size/1000000
            print("total number of model parameters in millions", model_size)
            avg_inference_time_single_batch=total_inference_time_single_batch/len(dataloader)
            print("average inference time in milliseconds per document using single batch size", avg_inference_time_single_batch)
            avg_memory_single_batch=total_memory_single_batch/len(dataloader)
            print("average peak inference memory in MB per document using single batch size", avg_memory_single_batch)
            
            if cfg["dataset_split"]=="train":
                gold_fname='/home/mila/x/xiyuan.zou/research/kd-coref/data/ontonotes_train.conll'
            elif cfg["dataset_split"]=="val":
                gold_fname = '/home/mila/x/xiyuan.zou/research/kd-coref/data/ontonotes_dev.conll'
            elif cfg["dataset_split"]=="test":
                gold_fname='/home/mila/x/xiyuan.zou/research/kd-coref/data/ontonotes_test.conll'
            pred_fname='/network/scratch/x/xiyuan.zou/kd-coref-project-output/results_file.conll'
            part_a=["perl", "/home/mila/x/xiyuan.zou/research/kd-coref/evaluators/reference-coreference-scorers/scorer.pl"]
            part_b=[gold_fname, pred_fname]
            kwargs={"capture_output": True, "check": True, "text": True}
            results={}
            for metric in ["muc", "bcub", "ceafe"]:
                r, p, f=extract_r_p_f(subprocess.run(part_a + [metric] + part_b, **kwargs))
                results[metric]={'recall': r, 'precision': p, 'f1': f}
            overall_f1=(results["muc"]["f1"]+results["bcub"]["f1"]+results["ceafe"]["f1"])/3
            print(results)
            print("Overall F1 is", overall_f1)
    
    
    #evaluate model on GAP by measuring raw F1
    if cfg["dataset_name"]=="GAP":
        id_list=[] #record the id and whether span A and B corefer for all docs
        Acoref_list=[]
        Bcoref_list=[]
        
        with torch.no_grad():
            for batch in iter(tqdm(dataloader)):
                GAP_gold_pairs=batch[-1]
                if cfg["inference_device"]=="gpu":
                    batch=[batch[i].cuda(0) for i in range(4)]+[batch[i] for i in range(4,7)]+[batch[i].cuda(0) for i in range(7,9)]
                else:
                    batch=[batch[i] for i in range(9)]
                assert len(batch[0])==1 #there should be only one segment
                output_list=model(*batch)
                top_span_starts, top_span_ends, top_antecedent_ids, top_antecedent_scores=output_list[3], output_list[4], output_list[5], output_list[6]
                top_span_starts, top_span_ends, top_antecedent_ids, top_antecedent_scores=top_span_starts.tolist(), top_span_ends.tolist(), top_antecedent_ids.tolist(), top_antecedent_scores.tolist()
                predicted_clusters, _, _=model.get_predicted_clusters(top_span_starts, top_span_ends, top_antecedent_ids, top_antecedent_scores)
                pronoun_subtoken_idx, A_subtoken_idx, B_subtoken_idx=GAP_gold_pairs[1], GAP_gold_pairs[2], GAP_gold_pairs[3]
                Acoref, Bcoref=False, False
                for cluster in predicted_clusters:
                    if pronoun_subtoken_idx in cluster and A_subtoken_idx in cluster:
                        Acoref=True
                    if pronoun_subtoken_idx in cluster and B_subtoken_idx in cluster:
                        Bcoref=True
                Acoref_list.append(Acoref)
                Bcoref_list.append(Bcoref)
                id_list.append(GAP_gold_pairs[0])
               
        with open('/network/scratch/x/xiyuan.zou/kd-coref-project-output/results_file.tsv', 'w', newline='') as tsvfile:
            fieldnames=['ID', 'A-coref', 'B-coref']
            writer=csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter='\t')
            for i in range(len(id_list)):
                writer.writerow({'ID': id_list[i], 'A-coref': Acoref_list[i], 'B-coref': Bcoref_list[i]})
        
        if cfg["dataset_split"]=="train":
            gold_tsv_path="/network/scratch/x/xiyuan.zou/kd-coref-project-output/gap-development.tsv"
        if cfg["dataset_split"]=="test":
            gold_tsv_path="/network/scratch/x/xiyuan.zou/kd-coref-project-output/gap-test.tsv"
        pred_tsv_path="/network/scratch/x/xiyuan.zou/kd-coref-project-output/results_file.tsv"
        scorecard=run_scorer(gold_tsv_path, pred_tsv_path)
        print(scorecard)
                    
    
if __name__ == '__main__':
    main()