import pandas as pd

"""
Explore the dataset (1.5 points)
Read the documentation (https://sites.google.com/view/cwisharedtask2018/datasets) of
the dataset and provide an answer to the following questions:

a) What do the start and offset values refer to? Provide an example.
b) What does it mean if a target word has a probabilistic label of 0.4?
c) The dataset was annotated by native and non-native speakers. How do the binary and
the probabilistic complexity label account for this distinction?
"""





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

data = pd.read_csv("../data/original/english/WikiNews_Train.tsv", sep='\t', header=None)





print("DONE")

"""Data desciption
Each line represents a sentence with one complex word annotation and relevant information, each separated by a TAB character.

    The first column shows the HIT ID of the sentence. All sentences with the same ID belong to the same HIT.
    The second column shows the actual sentence where there exists a complex phrase annotation.
    The third and fourth columns display the start and end offsets of the target word in this sentence.
    The fifth column represents the target word.
    The sixth and seventh columns show the number of native annotators and the number of non-native annotators who saw the sentence.
    The eighth and ninth columns show the number of native annotators and the number of non-native annotators who marked the target word as difficult.
    The tenth and eleventh columns show the gold-standard label for the binary and probabilistic classification tasks.

The labels in the binary classification task were assigned in the following manner:

    0: simple word (none of the annotators marked the word as difficult)
    1: complex word (at least one annotator marked the word as difficult)

The labels in the probabilistic classification task were assigned as <the number of annotators who marked the word as difficult>/<the total number of annotators>."""






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



# TODO: which data to use, preprocessed or raw?
# TODO: should we just compare the distributions of lengths/frequencies in two classes?
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




