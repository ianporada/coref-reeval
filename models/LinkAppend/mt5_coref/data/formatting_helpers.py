"""Helper functions for formatting or parsing io strings."""

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def normalize_speaker(speaker_in):
  """Add '_' before and after speaker name if it does not contain it already."""
  if not speaker_in or speaker_in == '-' or speaker_in == '__':
    return '_'

  speaker = speaker_in.replace(' ', '_')
  speaker = speaker.strip()

  if not speaker.startswith('_'):
    speaker = '_' + speaker

  if not speaker.endswith('_'):
    speaker = speaker + '_'

  return speaker


def split_action(action_str):
    """Parse an action str into a reference and an assignment."""
    mentions = [x.strip() for x in action_str.split('->')]
    
    if len(mentions) == 1:
        raise ValueError('action_str with one reference: "%s"' % action_str)
        # alternative would be to treat as a self-reference (i.e. for datasets with singletons)
        # reference = mentions[0]
        # return reference, reference
    
    reference, assignment = mentions
    return reference, assignment


def assignment_to_link(assignment_str):
    """Returns the cluster (int) of the link assignment in format "[%n", otherwise returns None."""
    if assignment_str[0] == '[':
        cluster_str = assignment_str[1:]
        if cluster_str.isdigit():
            return int(cluster_str)
    return None


def parse_actions(output_str):
    """Parse output as a list of actions where an action is a (reference, assignment) tuple."""
    action_strs = output_str.split(';;')
    
    link_actions = []
    append_actions = []
    
    for action in action_strs:
        if not action:
            continue
        
        try:
            reference, assignment = split_action(action)
        except ValueError as e:
            logging.info(e)
            continue
        
        assignment_link = assignment_to_link(assignment)
        if assignment_link is not None:
            link_actions.append((reference, assignment_link))
        else:
            append_actions.append((reference, assignment))
            
    return link_actions, append_actions


def find_nth_span_in_tokens(model_input, span, n, sentence_id=None):
    """Find the nth index of a sublist in a list."""
    count = 0
    for i in range(len(model_input.tokens) - 1, -1, -1):
        if sentence_id and model_input.token_map[i] and (model_input.token_map[i][0] != sentence_id):
            continue
        if model_input.tokens[i : i + len(span)] == span:
            count += 1
            if count == n:
                return i
    return -1


def clean_mention_part(tokens):
    """Remove everything after ** in a mention or context
    In reference code, ']]' might also end a mention.
    """
    if '**' in tokens:
        return tokens[:tokens.index('**') + 1]
    return tokens
