import spacy
import string
import pandas as pd
from collections import Counter

nlp = spacy.load("en_core_web_sm")

"""Tokenization
Process the dataset using the spaCy package and extract the following information:
Number of tokens:
Number of types:  
Number of words: (decide if lowercase or unique)
Average number of words per sentence:
Average word length: 
Provide the definition that you used to determine words: 
"""

with open("../data/preprocessed/train/sentences.txt") as f:
    text = f.read()

doc = nlp(text)

tokens_length = len(doc)

token_list = []
for token in doc:
    if token.text not in ["\n"]:
        token_list.append(token.text)

token_length = len(token_list)
types_length = len(list(set(token_list)))

# words defined as not punctuation
punctuations = list(string.punctuation)

# TODO: should we do lowercase?
# TODO: unique or total?

words = [x for x in token_list if x not in punctuations]
words_length = len(words)

# stopwords = nlp.Defaults.stop_words
# print(stopwords)


average_word_length = sum(len(word) for word in words) / len(words)

length_of_words = []
for sent in str(doc).split("\n"):
    word_list = sent.split(" ")
    words = [x for x in word_list if x not in punctuations]
    length_of_words.append(len(words))

av_no_words_sentence = sum(length_of_words) / (len(length_of_words))

# definition:  tokens that are not in punctuation from spacy

print("DONE")

# --------------------------------------------------------------------------------------------
""" Word Classes
Run the default part-of-speech tagger on the dataset and identify the ten most frequent 
POS tags. Complete the table below for these ten tags (the tagger in the model 
en_core_web_sm is trained on the PENN Treebank tagset).
"""

tmp_list = []
for token in doc:
    if str(token) not in ["\n"]:
        tmp_list.append({"token": token.text, "pos": token.pos_})

pos_df = pd.DataFrame(tmp_list)

# TODO: should we clean the data more? remove punct and \\? REMOVE

# group by token, pos_ and get the frequency
freq_df = pos_df[['pos']].value_counts().reset_index()
freq_df.columns = ["pos", "freq"]

token_freq_df = pos_df[['pos', 'token']].value_counts().reset_index()

# 'NOUN', 'PROPN', 'PUNCT', 'VERB', 'ADP', 'DET', 'ADJ', 'AUX', 'PRON', 'ADV'

# add relative freq

total_num = sum(list(freq_df['freq']))
freq_df['relative'] = freq_df['freq'] / total_num * 100
freq_df['relative'] = freq_df['relative'].apply(lambda x: round(x, 2))

freq_df.to_csv("alicja_tmp_files/pos_frequencies.csv")

# add 3 most frequent tokens with this tag

tags = ['NOUN', 'PROPN', 'PUNCT', 'VERB', 'ADP', 'DET', 'ADJ', 'AUX', 'PRON', 'ADV']

# TODO: should we clean the data more? remove punct and \\? remove punctuation

tags_data = []
for tag in tags:
    tmp_data = token_freq_df[token_freq_df['pos'] == tag]
    three_frequent = list(tmp_data['token'].iloc[0:3])
    one_infrequent = tmp_data['token'].iloc[-1]
    tags_data.append({"pos": tag, "freq": three_frequent, "infreq": one_infrequent})

final_tokens = pd.DataFrame(tags_data)
final_tokens.to_csv("alicja_tmp_files/pos_tokens.csv")

print("DONE")

# 10 most frequent tags

# Finegrained POS-tag # Universal POS-Tag # Occurrences # Relative Tag Frequency (%)
# 3 most frequent tokens # with this tag # Example for an infrequent token with this tag


# --------------------------------------------------------------------------------------------
""" N-Grams
Calculate the distribution of n-grams and provide the 3 most frequent
Token bigrams: 
Token trigrams:
POS bigrams:
POS trigrams:
"""


# TODO: should we clean data somehow? don't remove punctuation

def get_n_grams(sentences, n=2):
    total_list = []
    for sent in sentences:
        sent = sent.split()
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


input_sentences = doc.text.split("\n")

tokens_bigrams = get_n_grams(input_sentences, 2)
tokens_trigrams = get_n_grams(input_sentences, 3)

final_bigrams = get_most_freq(tokens_bigrams)
final_trigrams = get_most_freq(tokens_trigrams)

print("token bigrams", final_bigrams[0], final_bigrams[1], final_bigrams[3])
print("token trigrams", final_trigrams[0], final_trigrams[1], final_trigrams[3])

print("DONE")


def pos_grams(doc, input_tokens, n):
    input_tokens = [token.pos_ for token in doc if token.is_alpha]
    POS_gram_list = [input_tokens[i:i + n] for i in range(len(input_tokens) - n + 1)]
    return POS_gram_list


tokens = [token for token in doc if token.text != "\n"]
POS_bigrams = pos_grams(doc,tokens,2)
print(Counter(str(elem) for elem in POS_bigrams).most_common(3))

POS_trigrams = pos_grams(doc,tokens,3)
print(Counter(str(elem) for elem in POS_trigrams).most_common(3))


# --------------------------------------------------------------------------------------------
""" Lemmatization
Provide an example for a lemma that occurs in more than two inflections in the dataset. 
Lemma:
Inflected Forms: 
Example sentences for each form:
"""

# TODO: do we have to code it or just manually find?


# --------------------------------------------------------------------------------------------
""" Named Entity Recognition
Number of named entities:
Number of different entity labels:  
Analyze the named entities in the first five sentences. Are they identified correctly? If not, 
explain your answer and propose a better decision.
"""

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
    tmp_sent = nlp(sent).ents
    tmp_ner = []
    tmp_tokens = []
    for k in tmp_sent:
        tmp_ner.append(k.label_)
        tmp_tokens.append(k.text)

    tmp_list.append({"sent": sent, "tokens": tmp_tokens, "ner": tmp_ner})

final_ner = pd.DataFrame(tmp_list)
final_ner.to_csv("alicja_tmp_files/5_sentences_ner.csv")

print("DONE")



