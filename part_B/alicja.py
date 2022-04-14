import pandas as pd
import spacy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

"""
Explore the dataset (1.5 points)
Read the documentation (https://sites.google.com/view/cwisharedtask2018/datasets) of
the dataset and provide an answer to the following questions:

a) What do the start and offset values refer to? Provide an example.
b) What does it mean if a target word has a probabilistic label of 0.4?
c) The dataset was annotated by native and non-native speakers. How do the binary and
the probabilistic complexity label account for this distinction?
"""

df = pd.read_csv('../data/original/english/WikiNews_Train.tsv', sep='\t', header=None)
data = df.set_axis(
    ['id', 'sentence', 'start', 'end', 'trgt', 'nat_ann', 'nonnat_ann', 'nat_diff', 'non_diff', 'label', 'prob'],
    axis=1, inplace=False)

"""
Extract basic statistics (0.5 point)
Let’s have a closer look at the labels for this task. 
Use the file data/original/english/WikiNews_Train.tsv and extract the following columns: 
Target word, binary label, probabilistic label
Provide the following information:  

Number of instances labeled with 0: 
Number of instances labeled with 1: 
Min, max, median, mean, and stdev of the probabilistic label: 
Number of instances consisting of more than one token: 
Maximum number of tokens for an instance: 
"""

num_of_inst_0 = len(data[(data["label"] == 0)])
num_of_inst_1 = len(data[(data["label"] == 1)])

print(data.describe())

min_prob_label = data["prob"].min()
max_prob_label = data["prob"].max()
mean_prob_label = data["prob"].mean()
median_prob_label = data["prob"].median()
std_prob_label = data["prob"].std()

print("DONE")

nlp = spacy.load("en_core_web_sm")

num_inst_mor_one_token = 0

for row in range(len(data)):
    doc = nlp(data.loc[row, "trgt"])
    for np in doc.noun_chunks:
        if len(np) > 1:
            num_inst_mor_one_token += 1

print("Number of instances consisting of more than one token : ", num_inst_mor_one_token)  # 1624

max_num_of_tokens = 0
maximum_token = ''

for row in range(len(data)):
    doc = nlp(data.loc[row, "trgt"])
    for np in doc.noun_chunks:
        if len(np) > max_num_of_tokens:
            max_num_of_tokens = len(np)
            maximum_token = np.text

print("Maximum number of tokens for an instance : ", max_num_of_tokens)  # 7
print("The maximum token instance : ", maximum_token)  # state-owned RIA Novosti news agency

"""
Explore linguistic characteristics (2 points)
For simplicity, we will focus on the instances which consist only of a single token and 
have been labeled as complex by at least one annotator. 
Calculate the length of the tokens as the number of characters. 
Calculate the frequency of the tokens using the wordfreq package 
(https://pypi.org/project/wordfreq/). 

Provide the Pearson correlation of length and frequency with the probabilistic complexity 
label:
Pearson correlation length and complexity: 
Pearson correlation frequency and complexity:
Provide 3 scatter plots with the probabilistic complexity on the y-axis. 
X-axis: 1) Length 2) Frequency 3) POS tag 
Set the ranges of the x and y axes meaningfully. 

Plot 1: 
Plot 2: 
Plot 3: 

Interpret the results.
"""




"""
Reflection (1 Point)
Can you think of another linguistic characteristic that might have an influence on the 
perceived complexity of a word? Propose at least one and explain your choice in 2-4 
sentences.
"""

# TODO: which data to use, preprocessed or raw? all classes
# TODO: should we just compare the distributions of lengths/frequencies in two classes? ok
"""
Baselines (2 Points)
Implement four baselines for the task in TODO_baselines.py. 
Majority baseline: always assigns the majority class
Random baseline: randomly assigns one of the classes
Length baseline: determines the class based on a length threshold 
Frequency baseline: determines the class based on a frequency threshold

Test different thresholds and choose the one which yields the highest accuracy on the 
dev_data: 
Length threshold: 
Frequency threshold:  
Fill in the table below (round to two decimals!): 
Interpret the results in 2-3 sentences. 
Store the predictions in a way that allows you to calculate precision, recall, and F-
measure and fill the table in exercise 12. 
"""
