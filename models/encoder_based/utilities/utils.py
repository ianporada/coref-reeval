import sys
sys.path.append('/home/mila/x/xiyuan.zou/research/kd-coref')
import torch
from utilities.consts import PRONOUNS_GROUPS, STOPWORDS, CATEGORIES
import os
import re
import subprocess

def extract_r_p_f(proc: subprocess.CompletedProcess) -> float:
    """Extracts recall, precision, and f1"""
    prev_line = ""
    curr_line = ""
    for line in str(proc.stdout).splitlines():
        prev_line = curr_line
        curr_line = line
    text = prev_line
    
    scores = re.findall(r"\s*([0-9.]+)%", prev_line)
    return [float(s) for s in scores]

def flatten(l):
    return [item for sublist in l for item in sublist]

def mask_tensor(t, mask):
    t=t+((1.0-mask.float())*-10000.0)
    t=torch.clamp(t, min=-10000.0, max=10000.0)
    return t

def get_pronoun_id(span):
    if len(span) == 1:
        span = list(span)
        if span[0] in PRONOUNS_GROUPS:
            return PRONOUNS_GROUPS[span[0]]
    return -1

def get_category_id(mention, antecedent):
    mention, mention_pronoun_id = mention
    antecedent, antecedent_pronoun_id = antecedent

    if mention_pronoun_id > -1 and antecedent_pronoun_id > -1:
        if mention_pronoun_id == antecedent_pronoun_id:
            return CATEGORIES['pron-pron-comp']
        else:
            return CATEGORIES['pron-pron-no-comp']

    if mention_pronoun_id > -1 or antecedent_pronoun_id > -1:
        return CATEGORIES['pron-ent']

    if mention == antecedent:
        return CATEGORIES['match']

    union = mention.union(antecedent)
    if len(union) == max(len(mention), len(antecedent)):
        return CATEGORIES['contain']

    return CATEGORIES['other']

def bucket_distance(offsets):
    """ offsets: [num spans1, num spans2] """
    # 10 semi-logscale bin: 0, 1, 2, 3, 4, (5-7)->5, (8-15)->6, (16-31)->7, (32-63)->8, (64+)->9
    logspace_distance = torch.log2(offsets.to(torch.float)).to(torch.long) + 3
    identity_mask = (offsets <= 4).to(torch.long)
    combined_distance = identity_mask * offsets + (1 - identity_mask) * logspace_distance
    combined_distance = torch.clamp(combined_distance, 0, 9)
    return combined_distance


def batch_select(tensor, idx, device=torch.device('cpu')):
    """ select on the last axis according to the index """
    assert tensor.shape[0] == idx.shape[0]  # Same size of first dim
    dim0_size, dim1_size = tensor.shape[0], tensor.shape[1]
    tensor = torch.reshape(tensor, [dim0_size * dim1_size, -1])
    idx_offset = torch.unsqueeze(torch.arange(0, dim0_size, device=device) * dim1_size, 1)
    new_idx = idx + idx_offset
    selected = tensor[new_idx]
    if tensor.shape[-1] == 1:  # If selected element is scalar, restore original dim
        selected = torch.squeeze(selected, -1)
    return selected