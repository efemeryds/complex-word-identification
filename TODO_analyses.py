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

# first part
tmp_list = []
for token in doc:
    if token.text not in ["\n"] and token.text not in punctuations:
        tmp_list.append({"token": token.text, "pos": token.pos_})

pos_df = pd.DataFrame(tmp_list)
freq_df = pos_df[['pos']].value_counts().reset_index()
freq_df.columns = ["pos", "freq"]
token_freq_df = pos_df[['pos', 'token']].value_counts().reset_index()

total_num = sum(list(freq_df['freq']))
freq_df['relative'] = freq_df['freq'] / total_num * 100
freq_df['relative'] = freq_df['relative'].apply(lambda x: round(x, 2))

# second part
tags = ['NOUN', 'PROPN', 'VERB', 'ADP', 'DET', 'ADJ', 'PRON', 'AUX', 'ADV', 'NUM']

tags_data = []
for tag in tags:
    tmp_data = token_freq_df[token_freq_df['pos'] == tag]
    three_frequent = list(tmp_data['token'].iloc[0:3])
    one_infrequent = tmp_data['token'].iloc[-1]
    tags_data.append({"pos": tag, "freq": three_frequent, "infreq": one_infrequent})

final_tokens = pd.DataFrame(tags_data)

# POS tags. Complete the table below for these ten tags (the tagger in the model
# en_core_web_sm is trained on the PENN Treebank tagset).


print("DONE")


# 3. N-Grams (1.5 points)
# Calculate the distribution of n-grams and provide the 3 most frequent

def get_n_grams(sentences, n=2):
    total_list = []
    for sent in sentences:
        sent = str(sent).split()
        words_zip = zip(*[sent[i:] for i in range(n)])
        two_grams_list = [item for item in words_zip]
        total_list.append(two_grams_list)
    final = [item for sublist in total_list for item in sublist]
    return final


def get_most_freq(grams_list):
    count_freq = {}
    for item in grams_list:
        if item in count_freq:
            count_freq[item] += 1
        else:
            count_freq[item] = 1
    sorted_two_grams = sorted(count_freq.items(), key=lambda item: item[1], reverse=True)
    return sorted_two_grams


input_sentences = list(doc.sents)

# POS bigrams:
tokens_bigrams = get_n_grams(input_sentences, 2)
final_bigrams = get_most_freq(tokens_bigrams)
print("token bigrams", final_bigrams[0], final_bigrams[1], final_bigrams[3])

# POS trigrams:
tokens_trigrams = get_n_grams(input_sentences, 3)
final_trigrams = get_most_freq(tokens_trigrams)
print("token trigrams", final_trigrams[0], final_trigrams[1], final_trigrams[3])


# 4. Lemmatization (1 point)

# Provide an example for a lemma that occurs in more than two inflections in the dataset.

# Lemma:

# Inflected Forms:

# Example sentences for each form:


# TODO: Find manually

# 5. Named Entity Recognition (1 point)
# Number of named entities:
# Number of different entity labels:
# Analyze the named entities in the first five sentences. Are they identified correctly? If not,
# explain your answer and propose a better decision.

ner_labels = []

for token in doc.ents:
    ner_labels.append(token.label_)

print("DONE")

# number of named entities
print(len(ner_labels))

# number of different entity labels
print(len(list(set(ner_labels))))

# first 5 sentences
print(input_sentences[:5])

for token in doc.ents:
    ner_labels.append(token.label_)

tmp_list = []


for sent in input_sentences[:5]:
    tmp_sent = nlp(str(sent)).ents
    tmp_ner = []
    tmp_tokens = []
    for k in tmp_sent:
        tmp_ner.append(k.label_)
        tmp_tokens.append(k.text)

    tmp_list.append({"sent": sent, "tokens": tmp_tokens, "ner": tmp_ner})

final_ner = pd.DataFrame(tmp_list)
print("DONE")


