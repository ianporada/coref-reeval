import sys
sys.path.append('/home/mila/x/xiyuan.zou/research/kd-coref')
from stanza.utils.conll import CoNLL
import utilities.conll_transform as conll_transform

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
    conll_sentences = CoNLL.convert_dict(stanza_sentences)
    return conll_sentences


def add_coreference_column(conll_sents, cluster_id_to_spans):
    # get clusters as a list of list of MentionSpan
    clusters = list(cluster_id_to_spans.values())
    # convert each MentionSpan to an inclusive tuple (sent, start, end)
    #clusters = [[(span.sent_id, span.local_start, span.local_end) for span in cluster] for cluster in clusters]
    conll_transform.write_chains(conll_sents, clusters)
    return conll_sents

def write_docs_in_conll_format(docs, output_fname):
    """`docs` is a dict of the form {doc_id : (hf_sentences, cluster_id_to_spans)}
    where hf_sentences is the `sentences` object for the doc from the huggingface dataset (the `words` are written to the output file)
    and where cluster_id_to_spans is a dict {cluster_id : [(sent_id, start, end), ...]}"""
    conll_docs = {}
    for doc_id, doc_attributes in docs.items():
        hf_sentences, cluster_id_to_spans = doc_attributes
        conll_sentences = convert_huggingface_sentences_to_conll_format(hf_sentences)
        conll_sentences = add_coreference_column(conll_sentences, cluster_id_to_spans)
        conll_docs[doc_id] = conll_sentences
    conll_transform.write_file(output_fname, conll_docs)