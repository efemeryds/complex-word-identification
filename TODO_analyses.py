# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt

import spacy
import string
import pandas as pd
from collections import Counter

nlp = spacy.load("en_core_web_sm")
with open("data/preprocessed/train/sentences.txt") as f:
    text = f.read()

doc = nlp(text)

# 1. Tokenization (1 point)
# Process the dataset using the spaCy package and extract the following information:

token_list = []
for token in doc:
    if token.text not in ["\n"]:
        token_list.append(token.text)

# Number of tokens:
token_length = len(token_list)
print(token_length)

# Number of types:
types_length = len(list(set(token_list)))
print(types_length)

# Number of words:

# we compared this with spacy punctuation removal
punctuations = list(string.punctuation)

# total
words = [x for x in token_list if x not in punctuations]
words_length = len(words)
print(words_length)

# Average number of words per sentence:
av_no_words_sentence = words_length / (len(list(doc.sents)))


# Average word length:
average_word_length = sum(len(word) for word in words) / len(words)

# Provide the definition that you used to determine words:

"""
Word is something that is not in the punctuation from the string package. It is also not a new line. We don't
use lowercase and treat words with capital letter as separte ones. 
"""


# 2. Word Classes (1.5 points)
# Run the default part-of-speech tagger on the dataset and identify the ten most frequent




# POS tags. Complete the table below for these ten tags (the tagger in the model
# en_core_web_sm is trained on the PENN Treebank tagset).




print("DONE")

# 3



# 4




# 5








