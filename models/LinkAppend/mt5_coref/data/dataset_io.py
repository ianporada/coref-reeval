import json

import conll_transform


def split_doc_into_doc_parts(example):
    """take a doc and return the doc parts"""
    doc_parts_dict = {} # {0: [], 1:[], ...}
    for sent_dict in example['sentences'][0]:
        sent_part_id = sent_dict['part_id']
        if sent_part_id in doc_parts_dict:
            doc_parts_dict[sent_part_id].append(sent_dict)
        else:
            doc_parts_dict[sent_part_id] = [sent_dict]
    id = example['id'][0]
    return {'id': [f'{id}/part_{k}' for k in doc_parts_dict],
            'sentences': [doc_parts_dict[k] for k in doc_parts_dict]}
    
    
def conll_convert_dict(stanza_sentences):
    conll_output = []
    for sent in stanza_sentences:
        conll_sent = []
        for row in sent:
            word = row['text']
            if word in ['1)', '2)']:
                word = '-'
            conll_row = [str(row['id']), word] + ['_'] * 7 + ['-']
            conll_sent.append(conll_row)
        conll_output.append(conll_sent)
    return conll_output


def convert_huggingface_sentences_to_conll_format(hf_sentences):
    """HuggingFace sentences is a list of dictionaries, each with a 'words' key.
    E.g. [{words: ['Hi', '!']}, {words: ['How', 'are', 'you', '?']}]
    
    Conll format is a list of sentences where each sentence is a conll row where each row is a list of values.
    """
    # First converts to stanza format and uses the stanza library to convert to conll format.
    # Stanza sentences is a list of list of dictionaries, each with an 'id' and 'text' key.
    stanza_sentences = []
    for hf_sent in hf_sentences:
        stanza_sent = [{'id': i + 1, 'text': word} for i, word in enumerate(hf_sent['words'])]
        stanza_sentences.append(stanza_sent)
    conll_sentences = conll_convert_dict(stanza_sentences)
    return conll_sentences


def add_coreference_column(conll_sents, cluster_id_to_spans):
    # get clusters as a list of list of MentionSpan
    clusters = list(cluster_id_to_spans.values())
    # convert each MentionSpan to an inclusive tuple (sent, start, end)
    clusters = [[(span.sent_id, span.local_start, span.local_end) for span in cluster]
                for cluster in clusters]
    conll_transform.write_chains(conll_sents, clusters)
    return conll_sents


def read_jsonl(fname):
    with open(fname) as f:
        return [json.loads(line) for line in f]


def write_jsonl(fname, examples):
    with open(fname, 'w') as f:
            for ex in examples:
                json.dump(ex, f)
                f.write('\n')


def processors_to_conll_docs(finished_docs):
    conll_docs = {} # id to conll_sents
    for doc_processor in finished_docs:
        doc_id = doc_processor.id
        cluster_id_to_spans = doc_processor.cluster_id_to_spans
        hf_sentences = doc_processor.sentences
        
        conll_sents = convert_huggingface_sentences_to_conll_format(hf_sentences)
        conll_sents = add_coreference_column(conll_sents, cluster_id_to_spans)
        
        conll_docs[doc_id] = conll_sents
        
    return conll_docs
