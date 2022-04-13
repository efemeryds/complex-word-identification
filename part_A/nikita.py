# Part A
import spacy
nlp = spacy.load('en_core_web_sm')
sentences_path = "C:/Users/Nikita/OneDrive/Desktop/Course_Materials/NLP/Assignments/complex-word-identification/data/preprocessed/train/sentences.txt"
with open(sentences_path, encoding="utf8") as sentences:
    sentences_cont = sentences.read()
print(sentences_cont)
doc = nlp(sentences_cont)
# 1. Tokenization
# Process the dataset using the spaCy package and extract the following information:
# Number of tokens: 16130 or 15477 (not counting "\n") or (82147?)
count = 0
for token in doc:
    count += 1
print("Number of tokens with newline character: ", count)

tokens = [token for token in doc if token.text != "\n"]
num_tokens = len(tokens)
print("Number of tokens: ", num_tokens)

# Number of types: 3746 or 3745(not counting "\n")
types_list = []
for token in tokens:
    if token.text in types_list:
        continue
    else:
        types_list.append(token.text)

types = len(types_list)
print("Number of types: ", types)

# Number of words: 13242 (takes tokens without punctuation, words like "isn't" are count as 2 words)
# Number of words: 13122 (without punctuation and "'s" or "n't")
words = [token.text for token in doc if token.is_punct != True and token.text != '\n' and token.text != "'s" and token.text != "n't"]
num_words = len(words)
print("Number of words: ", num_words)

# Average number of words per sentence: 10.668292682926829 or 10 (or ~ 11)
# Number of sentences in doc: 1230
sent_count = 0
for sent in doc.sents:
    #print(sent)
    sent_count +=1
print("Number of sentences in doc: ", sent_count)
avg_words_per_sent = num_words/sent_count
print("Avg. number of words per sentence: ", avg_words_per_sent)
# Average word length:

# Provide the definition that you used to determine words:

# 2. Word Classes
POS_tag_list = []
for token in doc:
    POS_tag_list.append(token.pos_)

tag_list =[]
for token.pos_ in POS_tag_list:
    if token.pos_ in tag_list:
        continue
    else:
        tag_list.append(token.pos_)

#print(tag_list)
x=0
POS_count = []
while x <= len(tag_list):
    for token.pos_ in tag_list:
        POS_count.append((token.pos_, POS_tag_list.count(token.pos_)))
        x += 1
#print(len(POS_count))
POS_count.sort(reverse=True, key=lambda x:x[1])
print(POS_count)
# 3. N-Grams

# 4. Lemmatization

# 5. Named Entity Recognition(NER)


# Part B
