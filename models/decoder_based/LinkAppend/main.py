"""
Run inference on LinkAppend MT5
or generate training data for training a new LinkAppend model.
"""

import json
import logging
import os
import time

import click
import torch
from data.dataset_io import (add_coreference_column,
                             convert_huggingface_sentences_to_conll_format,
                             processors_to_conll_docs,
                             split_doc_into_doc_parts, write_jsonl)
from data.document_processor import DocumentProcessor
from datasets import load_dataset
from transformers import (DataCollatorWithPadding, MT5Tokenizer,
                          MT5ForConditionalGeneration)
import conll_transform

# TODO: worker multiprocessing? import multiprocessing

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def batch_model_input_to_output(tokenizer, model, input_tok_ids, max_new_tokens, collator_fn):
    features = [{'input_ids': x[0]} for x in input_tok_ids]
    batched_inputs = collator_fn(features)
    for k, v in batched_inputs.items():
        batched_inputs[k] = v.to('cuda')
        
    start = time.time()
        
    with torch.no_grad():
        generated_ids = model.generate(**batched_inputs, max_new_tokens=max_new_tokens)
        
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    inference_time = time.time() - start
    
    if len(generated_texts) == 1:
        logger.info('Model output: "%s"' % generated_texts)
    
    return generated_texts, inference_time


@click.command() # TODO: move config to hydra?
@click.option('--tokenizer_path', default='google/mt5-xxl')
@click.option('--model_path',     default='scratch/mt5_coref/mt5') # set to 'oracle' to generate the training set
@click.option('--output_dir',     default='output')
@click.option('--split',          default='train')
@click.option('--batch_size',     default=1, type=int)
@click.option('--max_input_size', default=2048, type=int)
@click.option('--max_new_tokens', default=384, type=int)
@click.option('--subset', default=0, type=int) # calculate subset of documents
@click.option('--subset_start', default=0, type=int) # calculate subset of documents
@click.option('--no_pound_symbol', is_flag=True)
@click.option('--dataset_name',        default='conll2012')
def main(tokenizer_path, model_path, output_dir, split, batch_size, max_input_size, max_new_tokens, subset, subset_start,
         no_pound_symbol, dataset_name):
    """Run inference using the MT5 shift-reduce model."""

    logger.info('Loading dataset.')
    if dataset_name == 'ontogum':
        dataset = load_dataset('coref-data/gum_indiscrim', 'ontogum')
        documents = dataset[split]
    elif dataset_name == 'gum':
        dataset = load_dataset('coref-data/gum_indiscrim', 'original')
        documents = dataset[split]
    elif dataset_name == 'arrau':
        dataset = load_dataset('coref-data/arrau_indiscrim')
        documents = dataset[split]
    elif dataset_name == 'litbank':
        dataset = load_dataset('coref-data/litbank_indiscrim', 'split_0')
        documents = dataset[split]
    elif dataset_name == 'preco':
        dataset = load_dataset('coref-data/preco_indiscrim')
        documents = dataset[split]
    else:
        dataset = load_dataset('coref-data/conll2012_indiscrim', 'english_v4')
        documents = dataset[split]

    documents = documents.sort("id")
    
    if subset > 0:
        documents = documents.select(list(range(subset_start, subset)))
        
    logger.info('Total number of document parts: %d' % len(documents))
    
    logger.info('Loading tokenizer and model.')
    tokenizer = MT5Tokenizer.from_pretrained(tokenizer_path, legacy=False) # use updated, non-legacy tokenizer
    collator_fn = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest', return_tensors="pt")
    
    use_oracle_model = (model_path == 'oracle')
    model = None
    if not use_oracle_model:
        model = MT5ForConditionalGeneration.from_pretrained(model_path)
        model = model.to(device='cuda')
        model.eval()
    
    saved_examples = [] # save all input/output pairs
    
    # Dataloading for the shift-reduce parser.
    docs_to_process = [x for x in documents]
    active_doc_processors = []
    finished_docs = []
    
    total_inference_time = 0.0
    
    # process all documents
    while active_doc_processors or docs_to_process:
        
        # process batch_size docs at a time
        while len(active_doc_processors) < batch_size and docs_to_process:
            next_document = docs_to_process.pop()
            document_processor = DocumentProcessor(next_document, tokenizer, max_input_size, no_pound_symbol)
            active_doc_processors.append(document_processor)
        
        # TODO: batch input to output mapping
        inputs = [x.get_next_model_input() for x in active_doc_processors]
        if use_oracle_model:
            outputs = [x.get_gold_output() for x in active_doc_processors]
        else:
            outputs, inference_time = batch_model_input_to_output(tokenizer, model, inputs, max_new_tokens, collator_fn)
            total_inference_time += inference_time
        
        # save the input/output pairs
        for i, input in enumerate(inputs):
            output_string = outputs[i]
            input_string = tokenizer.decode(input[0],
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
            example = {
                'input': input_string, 'output': output_string
            }
            saved_examples.append(example)
        
        for document_processor, output in zip(active_doc_processors, outputs):
            document_processor.update_context(output)
        
        finished_processors = [x for x in active_doc_processors if x.is_finished()]
        finished_docs += [x for x in finished_processors]
        
        if finished_processors:
            logging.info('Finished docs (%d total): %s' % (len(finished_docs), [x.id for x in finished_processors]))
        
        active_doc_processors = [x for x in active_doc_processors if not x.is_finished()]
        
    logging.info('Saving examples to jsonl file.')
    
    model_size = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
    model_name = str(model_size + '_' + os.path.basename(model_path)).replace('/', '_')
    
    if subset > 0:
        dataset_name = f'{dataset_name}-{subset_start}-{subset}'
    
    if no_pound_symbol:
        dataset_name = f'{dataset_name}-nopound'
    
    jsonl_fname = os.path.join(output_dir, f'{split}_{model_name}_{dataset_name}_examples.jsonl')
    write_jsonl(jsonl_fname, saved_examples)

    logging.info('Writing documents to conll file.')
    
    conll_docs = processors_to_conll_docs(finished_docs)
    
    conll_fname = os.path.join(output_dir, f'{split}_{model_name}_{dataset_name}_pred.conll')
    conll_transform.write_file(conll_fname, conll_docs)
    
    print('total time: ', total_inference_time)
    print('max memory: ', torch.cuda.max_memory_allocated())
    print(conll_fname)


if __name__ == '__main__':
    main()
