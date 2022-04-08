import spacy
import string
nlp = spacy.load("en_core_web_sm")

"""Tokenization
Process the dataset using the spaCy package and extract the following information:
Number of tokens:
Number of types:  
Number of words:
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
    token_list.append(str(token))

types_length = len(list(set(token_list)))

# words defined as not punctuation
punctuations = list(string.punctuation)

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

print("DONE")

""" Word Classes """






""" N-Grams """







""" Lemmatization """







""" Named Entity Recognition """













