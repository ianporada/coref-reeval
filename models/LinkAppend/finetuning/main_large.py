
import logging
import os
from functools import partial

import click
import datasets
import pandas as pd
import torch
import wandb
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, T5ForConditionalGeneration)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def show_samples(dataset, num_samples=3, seed=42):
    sample = dataset["train"].shuffle(seed=seed).select(range(num_samples))
    for example in sample:
        print(f"\n'>> Input: {example['input']}'")
        print(f"'>> Output: {example['output']}'")


def preprocess_function(examples, tokenizer, max_input_length, max_target_length):
    model_inputs = tokenizer(
        examples['input'],
        max_length=max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        examples['output'], max_length=max_target_length, truncation=True
    )
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


@click.command()
@click.option('--batch_size',     default=2, type=int)
@click.option('--gradient_accumulation_steps',     default=64, type=int)
def main(batch_size, gradient_accumulation_steps):
    """Run inference using the MT5 shift-reduce model."""
    # params
    model_path = 'google/mt5-large'
    max_input_length = 2048 # 2048
    max_target_length = 384 # 384
    
    # training params
    # batch_size = 32 # 8
    # gradient_accumulation_steps = 64 # 16 # 128 / batch_size
    # num_train_steps = 50_000 80 epochs, 3 epochs
    logging_steps = 250
    eval_steps = 2_000
    num_workers = 8
    optim="adafactor"
    
    # files
    model_name = model_path.split("/")[-1]
    scratch_dir = os.getenv('SCRATCH')
    dataset_name = 'llm-coref/mt5-coref-ontonotes' # 75,187 training examples
    output_dir = os.path.join(scratch_dir, 'mt5_checkpoints', model_name)
    model_output_dir = os.path.join(output_dir, 'output')
    logging_dir = os.path.join(output_dir, 'logs')
    wandb_dir = output_dir
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    wandb.init(project='seq2seq_coref', name=model_name, dir=wandb_dir)
    
    # Finetune the model
    
    dataset_dict = datasets.load_dataset(dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
    
    partial_preprocess_function = partial(preprocess_function,
                                          tokenizer=tokenizer,
                                          max_input_length=max_input_length,
                                          max_target_length=max_target_length)
    
    tokenized_datasets = dataset_dict.map(partial_preprocess_function, batched=True)
    
    show_samples(tokenized_datasets, num_samples=1)
    
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    args = Seq2SeqTrainingArguments(
        output_dir=model_output_dir,
        logging_dir=logging_dir,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=1.0e-3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_total_limit=40,
        load_best_model_at_end=True, # save best model
        # max_steps=num_train_steps,
        num_train_epochs=80,
        predict_with_generate=True,
        logging_steps=logging_steps,
        logging_first_step=True,
        # save_steps=eval_steps,
        # eval_steps=eval_steps,
        dataloader_num_workers=num_workers,
        bf16=True,
        optim=optim,
        # resume_from_checkpoint=True,
        run_name=model_name,
        lr_scheduler_type='constant'
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    tokenized_datasets = tokenized_datasets.remove_columns(
        dataset_dict['train'].column_names
    )
    
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    trainer.train(resume_from_checkpoint=True)


if __name__ == '__main__':
    main()
