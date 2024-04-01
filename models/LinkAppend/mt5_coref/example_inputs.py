
document = {
    'id': 'w',
    'sentences': [
        {
            'speaker': None,
            'words': ['The', 'Eiffel', 'Tower', '(', 'French', ':', 'tour', 'Eiffel', ')', 'is', 'a', 'wrought', '-', 'iron', 'lattice', 'tower', 'on', 'the', 'Champ', 'de', 'Mars', 'in', 'Paris', ',', 'France', '.', 'It', 'is', 'named', 'after', 'the', 'engineer', 'Gustave', 'Eiffel', ',', 'whose', 'company', 'designed', 'and', 'built', 'the', 'tower', '.']
        },
        {
            'speaker': None,
            'words': ['Locally', 'nicknamed', '"', 'La', 'dame', 'de', 'fer', '"', '(', 'French', 'for', '"', 'Iron', 'Lady', '"),', 'it', 'was', 'constructed', 'from', '1887', 'to', '1889', 'as', 'the', 'centerpiece', 'of', 'the', '1889', 'World', "'", 's', 'Fair', '.']
        },
        {
            'speaker': None,
            'words': ['Although', 'initially', 'criticised', 'by', 'some', 'of', 'France', "'", 's', 'leading', 'artists', 'and', 'intellectuals', 'for', 'its', 'design', ',', 'it', 'has', 'since', 'become', 'a', 'global', 'cultural', 'icon', 'of', 'France', 'and', 'one', 'of', 'the', 'most', 'recognisable', 'structures', 'in', 'the', 'world', '.']
        },
        {
            'speaker': None,
            'words': ['The', 'Eiffel', 'Tower', 'is', 'the', 'most', 'visited', 'monument', 'with', 'an', 'entrance', 'fee', 'in', 'the', 'world', ':', '6', '.', '91', 'million', 'people', 'ascended', 'it', 'in', '2015', '.']
        },
        {
            'speaker': None,
            'words': ['It', 'was', 'designated', 'a', 'monument', 'historique', 'in', '1964', ',', 'and', 'was', 'named', 'part', 'of', 'a', 'UNESCO', 'World', 'Heritage', 'Site', '("', 'Paris', ',', 'Banks', 'of', 'the', 'Seine', '")', 'in', '1991', '.']
        }
    ]
}

model_inputs = [
    [
        'coref: w | # _ The Eiffel Tower ( French : tour Eiffel ) is a wrought - iron lattice tower on the Champ de Mars in Paris , France . It is named after the engineer Gustave Eiffel , whose company designed and built the tower . ** # _ Locally nicknamed " La dame de fer " ( French for " Iron Lady "), it was constructed from 1887 to 1889 as the centerpiece of the 1889 World \' s Fair . # _ Although initially criticised by some of France \' s leading artists and intellectuals for its design , it has since become a global cultural icon of France and one of the most recognisable structures in the world . # _ The Eiffel Tower is the most visited monument with an entrance fee in the world : 6 . 91 million people ascended it in 2015 . # _ It was designated a monument historique in 1964 , and was named part of a UNESCO World Heritage Site (" Paris , Banks of the Seine ") in 1991 .'
    ],
    [
        'coref: w # _ [1 The Eiffel Tower ( French : tour Eiffel ) ] is a wrought - iron lattice tower on the Champ de Mars in Paris , France . [1 [2 It ] ] is named after the engineer Gustave Eiffel , whose company designed and built [2 the tower ] . | # _ Locally nicknamed " La dame de fer " ( French for " Iron Lady "), it was constructed from 1887 to 1889 as the centerpiece of the 1889 World ' s Fair . ** # _ Although initially criticised by some of France ' s leading artists and intellectuals for its design , it has since become a global cultural icon of France and one of the most recognisable structures in the world . # _ The Eiffel Tower is the most visited monument with an entrance fee in the world : 6 . 91 million people ascended it in 2015 . # _ It was designated a monument historique in 1964 , and was named part of a UNESCO World Heritage Site (" Paris , Banks of the Seine ") in 1991 .'
        'coref: w # _ [1 The Eiffel Tower ( French : tour Eiffel ) ] is a wrought - iron lattice tower on the Champ de Mars in Paris , France . [1 It ] is named after the engineer Gustave Eiffel , whose company designed and built [1 the tower ] . | # _ Locally nicknamed " La dame de fer " ( French for " Iron Lady "), it was constructed from 1887 to 1889 as the centerpiece of the 1889 World \' s Fair . ** # _ Although initially criticised by some of France \' s leading artists and intellectuals for its design , it has since become a global cultural icon of France and one of the most recognisable structures in the world . # _ The Eiffel Tower is the most visited monument with an entrance fee in the world : 6 . 91 million people ascended it in 2015 . # _ It was designated a monument historique in 1964 , and was named part of a UNESCO World Heritage Site (" Paris , Banks of the Seine ") in 1991 .'
    ],
    [
        'coref: w # _ [1 The Eiffel Tower ( French : tour Eiffel ) ] is a wrought - iron lattice tower on the Champ de Mars in Paris , France . [1 It ] is named after the engineer Gustave Eiffel , whose company designed and built [1 the tower ] . # _ Locally nicknamed " La dame de fer " ( French for " Iron Lady "), [1 it ] was constructed from 1887 to 1889 as the centerpiece of the 1889 World \' s Fair . | # _ Although initially criticised by some of France \' s leading artists and intellectuals for its design , it has since become a global cultural icon of France and one of the most recognisable structures in the world . ** # _ The Eiffel Tower is the most visited monument with an entrance fee in the world : 6 . 91 million people ascended it in 2015 . # _ It was designated a monument historique in 1964 , and was named part of a UNESCO World Heritage Site (" Paris , Banks of the Seine ") in 1991 .'
    ],
    [
        'coref: w # _ [1 The Eiffel Tower ( French : tour Eiffel ) ] is a wrought - iron lattice tower on the Champ de Mars in Paris , [2 France ] . [1 It ] is named after the engineer Gustave Eiffel , whose company designed and built [1 the tower ] . # _ Locally nicknamed " La dame de fer " ( French for " Iron Lady "), [1 it ] was constructed from 1887 to 1889 as the centerpiece of the 1889 World \' s Fair . # _ Although initially criticised by some of [2 France \' s ] leading artists and intellectuals for [1 its ] design , [1 it ] has since become a global cultural icon of [2 France ] and one of the most recognisable structures in the world . | # _ The Eiffel Tower is the most visited monument with an entrance fee in the world : 6 . 91 million people ascended it in 2015 . ** # _ It was designated a monument historique in 1964 , and was named part of a UNESCO World Heritage Site (" Paris , Banks of the Seine ") in 1991 .'
    ],
    [
        'coref: w # _ [1 The Eiffel Tower ( French : tour Eiffel ) ] is a wrought - iron lattice tower on the Champ de Mars in Paris , [2 France ] . [1 It ] is named after the engineer Gustave Eiffel , whose company designed and built [1 the tower ] . # _ Locally nicknamed " La dame de fer " ( French for " Iron Lady "), [1 it ] was constructed from 1887 to 1889 as the centerpiece of the 1889 World \' s Fair . # _ Although initially criticised by some of [2 France \' s ] leading artists and intellectuals for [1 its ] design , [1 it ] has since become a global cultural icon of [2 France ] and one of the most recognisable structures in [3 the world ] . # _ [1 The Eiffel Tower ] is the most visited monument with an entrance fee in [3 the world ] : 6 . 91 million people ascended [1 it ] in 2015 . | # _ It was designated a monument historique in 1964 , and was named part of a UNESCO World Heritage Site (" Paris , Banks of the Seine ") in 1991 . **'
    ],
]

model_outputs = [
    [
       'It ## is named after -> The Eiffel Tower ( French : tour Eiffel ) ## is a wrought ;; the tower ## . ** -> It ## is named after ;;'
     # 'It ## is named after -> The Eiffel Tower ( French : tour Eiffel ) ## is a wrought ;; the tower ## . ** _ -> It ## is named after ;;'
    ],
    [
        'it ## was constructed from -> [1 ;;'
    ],
    [
        'its ## design , it -> [1 ;; it ## has since become -> its ## design , it ;; France \' s ## leading artists and -> France ## . [1 It ;; France ## and one of -> France \' s ## leading artists and ;;'
    ],
    [
        'The Eiffel Tower ## is the most -> [1 ;; it ## in 2015 . -> The Eiffel Tower ## is the most ;; the world ## : 6 . -> the world ## . | # ;;'
    ],
    [
        'It ## was designated a -> [1 ;; Paris ## , Banks of -> Paris , [2 France ## ] . [1 ;;'
    ],
]

input_to_output = {}
for i, input in enumerate(model_inputs):
    input_to_output[input[0]] = model_outputs[i][0]

def convert_input_to_output(input):
    return input_to_output[input]
