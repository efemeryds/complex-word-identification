import spacy
import pandas as pd
import string
from collections import Counter

nlp = spacy.load("en_core_web_sm")
punctuations = string.punctuation

df_file = open("../datas/sentences.txt", encoding='utf-8')
data = df_file.read()
doc = nlp(data)


def tokenization(text):
    token_list = []

    for token in text:
        token_list.append(token.text)

    return token_list


def num_of_tokens(text):
    total_token = 0
    for token in text:
        total_token += 1

    return total_token


def num_of_types(text):
    """
    Type is different from the number of actual occurrences which would be known as tokens.
    """

    token_list = set(tokenization(text))
    return len(token_list)


def num_of_words(text):
    """
    A word is a speech sound or a combination of sounds, or its representation in writing,
    that symbolizes and communicates a meaning and may consist of a single morpheme or a combination of morphemes.
    """

    total_words = 0
    words = []

    for token in text:
        if token.text not in punctuations and token.text != '\n':
            total_words += 1
            words.append(token.text)

    return total_words


def list_of_words(text):
    """
    A word is a speech sound or a combination of sounds, or its representation in writing,
    that symbolizes and communicates a meaning and may consist of a single morpheme or a combination of morphemes.
    """

    total_words = 0
    words = []

    for token in text:
        if token.text not in punctuations and token.text != '\n':
            total_words += 1
            words.append(token.text)

    return words



def average_num_of_words_per_sentence(text):
    sentence_count = 0

    for sent in text.sents:
        sentence_count += 1

    avg_num_words = (num_of_words(text)) / sentence_count
    print(num_of_words(text), sentence_count)

    return round(avg_num_words, 2)  # 23.63 number of words per sentence


def average_word_length(text):
    words_list = list_of_words(text)
    avg_num_words = 0

    avg_num_words = sum(len(word) for word in words_list) / len(words_list)

    return round(avg_num_words, 2)  # 4.9 average word length


# 2. Word Classes
tag_frequencies = Counter()

for sentence in doc.sents:
    tags = []
    for token in sentence:
        # if not token.is_punct:
        tags.append(token.tag_)
    tag_frequencies.update(tags)

print(tag_frequencies)


token_frequencies = Counter()

for sentence in doc.sents:
    tokens = []
    token_list = ['NN','NNP','IN','DT','JJ','NNS', ',','VBD','.','_SP','VBN','RB','CD']
    for token in sentence:
        if token.tag_ in token_list:
            tokens.append((token.tag_,token.text))
    token_frequencies.update(tokens)


taglist_frequencies = Counter()

for sentence in doc.sents:
    tag_list = []
    for token in sentence:
        tag_list.append(token.tag_)
    taglist_frequencies.update(tag_list)

#print(tag_list)
#print(taglist_frequencies)


counts = [(token.tag_, tag_list.count(token.tag_) / len(taglist_frequencies)) for token.tag_ in set(token_list)]


# N-grams
def n_grams(text, tokens, n):
    tokens = [token.text for token in text]

    return [tokens[i:i + n] for i in range(len(tokens) - n + 1)]


def most_frequent(List):
    counter = 0
    item = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            item = i

    return item

# 4.Lemmatization

def lemmatization(text):
    for token in text:
        if token.pos_ == 'VERB':
            print('{} -> {}'.format(token, token.lemma_))


# 5. Named Entity Recognition
def ner(text):
    list_of_ent = []

    for ent in text.ents:
        list_of_ent.append(ent.label_)
        print(ent.text, ent.label_)
    print(len(list_of_ent))  # 1627 labels


def ner_different(text):
    list_of_diff_ent = []

    for ent in text.ents:
        list_of_diff_ent.append(ent.label_)
    print(len(set(list_of_diff_ent)))


# Analyze the named entities in the first five sentences. Are they identified correctly?

def first_five_sentence(text):
    first_five_sentence = []
    for sentence in doc.sents:
        first_five_sentence.append(sentence)

    return (first_five_sentence[:5])

first_five_sentence_list = str(first_five_sentence(doc))
ner_first_five = []
doc_5 = nlp(first_five_sentence_list)
for ent in doc_5.ents:
    ner_first_five.append((ent.text, ent.label_))

print(ner_first_five)
print(len(ner_first_five))