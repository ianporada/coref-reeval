from enum import Enum

SPEAKER_START = 49518 
SPEAKER_END = 22560 
NULL_ID_FOR_COREF = 0

PRONOUNS_GROUPS = {
            'i': 0, 'me': 0, 'my': 0, 'mine': 0, 'myself': 0,
            'you': 1, 'your': 1, 'yours': 1, 'yourself': 1, 'yourselves': 1,
            'he': 2, 'him': 2, 'his': 2, 'himself': 2,
            'she': 3, 'her': 3, 'hers': 3, 'herself': 3,
            'it': 4, 'its': 4, 'itself': 4,
            'we': 5, 'us': 5, 'our': 5, 'ours': 5, 'ourselves': 5,
            'they': 6, 'them': 6, 'their': 6, 'themselves': 6,
            'that': 7, 'this': 7
            }

STOPWORDS = {"'s", 'a', 'all', 'an', 'and', 'at', 'for', 'from', 'in', 'into',
             'more', 'of', 'on', 'or', 'some', 'the', 'these', 'those'}

CATEGORIES = {'pron-pron-comp': 0,
              'pron-pron-no-comp': 1,
              'pron-ent': 2,
              'match': 3,
              'contain': 4,
              'other': 5
              }


class Gender(Enum):
  UNKNOWN = 0
  MASCULINE = 1
  FEMININE = 2


# Mapping of (lowercased) pronoun form to gender value. Note that reflexives
# are not included in GAP, so do not appear here.
PRONOUNS = {
    'she': Gender.FEMININE,
    'her': Gender.FEMININE,
    'hers': Gender.FEMININE,
    'he': Gender.MASCULINE,
    'his': Gender.MASCULINE,
    'him': Gender.MASCULINE,
}

# Fieldnames used in the gold dataset .tsv file.
GOLD_FIELDNAMES = [
    'ID', 'Text', 'Pronoun', 'Pronoun-offset', 'A', 'A-offset', 'A-coref', 'B',
    'B-offset', 'B-coref', 'URL'
]

# Fieldnames expected in system output .tsv files.
SYSTEM_FIELDNAMES = ['ID', 'A-coref', 'B-coref']