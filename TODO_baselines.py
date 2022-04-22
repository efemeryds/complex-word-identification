# Implement four baselines for the task.
# Majority baseline: always assigns the majority class of the training data
# Random baseline: randomly assigns one of the classes. Make sure to set a random seed and average the accuracy over 100 runs.
# Length baseline: determines the class based on a length threshold
# Frequency baseline: determines the class based on a frequency threshold
import pandas as pd

from model.data_loader import DataLoader
from collections import Counter
from sklearn.metrics import accuracy_score
import random
import spacy
from wordfreq import zipf_frequency, word_frequency
nlp = spacy.load("en_core_web_sm")

# Each baseline returns predictions for the test data. The length and frequency baselines determine a threshold using the development data.

def majority_baseline(train_sentences_input, train_labels_input, test_input_input, test_labels_input):

    train_labels_list = []
    for element in train_labels_input:
        train_labels_list.append(element.strip())

    train_labels_list = " ".join(train_labels_list).split(" ")
    n_occurences = train_labels_list.count("N")
    c_occurences = train_labels_list.count("C")

    if n_occurences > c_occurences:
        majority_class = "N"
    else:
        majority_class = "C"

    predictions = []
    tokens_list = []
    for instance in test_input_input:
        tokens = instance.split(" ")
        instance_predictions = [majority_class for t in tokens]
        predictions.append(instance_predictions)
        tokens_list.append(tokens)

    final_predictions = [item for sublist in predictions for item in sublist]
    final_tokens = [item for sublist in tokens_list for item in sublist]

    test_labels_tmp = []
    for element in test_labels_input:
        test_labels_tmp.append(element.strip())
    test_labels_list = " ".join(test_labels_tmp).split(" ")

    final_df = pd.DataFrame({"tokens": final_tokens, "predictions": final_predictions})

    accuracy = accuracy_score(test_labels_list, final_predictions)
    return accuracy, final_df


def random_baseline(train_sentences, train_labels, test_input, test_labels):
    subjects = ["N", "C"]

    predictions = []
    tokens_list = []
    for instance in test_input:
        tokens = instance.split(" ")
        instance_predictions = [random.choice(subjects) for t in tokens]
        predictions.append(instance_predictions)
        tokens_list.append(tokens)

    predictions = [item for sublist in predictions for item in sublist]
    final_tokens = [item for sublist in tokens_list for item in sublist]

    test_labels_list = []
    for element in test_labels:
        test_labels_list.append(element.strip())
    test_labels_list = " ".join(test_labels_list).split(" ")

    final_df = pd.DataFrame({"tokens": final_tokens, "predictions": predictions})
    accuracy = accuracy_score(test_labels_list, predictions)
    return accuracy, final_df


def length_baseline(train_sentences, train_labels, test_input_input, test_labels_input):
    predictions = []

    token_length_each = []
    tokens_each = []

    for instance in train_sentences:
        doc = nlp(instance)
        for token in doc:
            if token.text not in ["\n"]:
                tokens_each.append(token.text)
                token_length_each.append(len(token.text))

    # length > 7 "C", length < 7 "N"
    predictions = []
    tokens_list = []
    for instance in test_input_input:
        tokens = instance.split(" ")
        for k in tokens:
            if k not in ["\n"]:
                length = len(k)
                if length > 7:
                    label = "C"
                else:
                    label = "N"

                predicted_label = label
                predictions.append(predicted_label)
                tokens_list.append(k)

    test_labels_list = []
    for element in test_labels_input:
        test_labels_list.append(element.strip())
    test_labels_list = " ".join(test_labels_list).split(" ")

    final_df = pd.DataFrame({"tokens": tokens_list, "predictions": predictions})
    accuracy = accuracy_score(test_labels_list, predictions)

    return accuracy, final_df


def frequency_baseline(train_sentences, train_labels, test_input_input, test_labels_input):
    frequency_of_tokens = {}
    freq_list = []

    for instance in train_sentences:
        doc = nlp(instance)
        for token in doc:
            if token.text not in ["\n"]:
                frequency = zipf_frequency(token.text, 'en', wordlist='small')
                frequency_of_tokens.update({token.text: frequency})
                freq_list.append(frequency)

    # > 6 N; <6 C
    predictions = []
    tokens_list = []
    for instance in test_input_input:
        tokens = instance.split(" ")
        for k in tokens:
            if k not in ["\n"]:
                try:
                    token_freq = freq_list[k]
                    if token_freq > 6:
                        label = "N"
                    else:
                        label = "C"
                except:
                    label = "N"
                predicted_label = label
                predictions.append(predicted_label)
                tokens_list.append(k)

    test_labels_list = []
    for element in test_labels_input:
        test_labels_list.append(element.strip())
    test_labels_list = " ".join(test_labels_list).split(" ")

    final_df = pd.DataFrame({"tokens": tokens_list, "predictions": predictions})
    accuracy = accuracy_score(test_labels_list, predictions)
    return accuracy, final_df


if __name__ == '__main__':
    train_path = "data/preprocessed/train/"
    dev_path = "data/preprocessed/val/"
    test_path = "data/preprocessed/test/"

    # Note: this loads all instances into memory. If you work with bigger files in the future, use an iterator instead.

    with open(train_path + "sentences.txt") as sent_file:
        train_sentences = sent_file.readlines()

    with open(train_path + "labels.txt") as label_file:
        train_labels = label_file.readlines()

    with open(dev_path + "sentences.txt") as dev_file:
        dev_sentences = dev_file.readlines()

    with open(dev_path + "labels.txt") as dev_label_file:
        dev_labels = dev_label_file.readlines()

    with open(test_path + "sentences.txt") as testfile:
        test_input = testfile.readlines()

    with open(test_path + "labels.txt") as test_label_file:
        test_labels = test_label_file.readlines()

    majority_accuracy, majority_predictions = majority_baseline(train_sentences, train_labels, test_input, test_labels)
    majority_accuracy_dev, majority_predictions_dev = majority_baseline(train_sentences, train_labels, dev_sentences,
                                                                        dev_labels)

    print("Test acuracy for majority test:", majority_accuracy)
    print("Test acuracy for majority dev:", majority_accuracy_dev)

    random_accuracy, random_predictions = random_baseline(train_sentences, train_labels, test_input, test_labels)
    random_accuracy_dev, random_predictions_dev = random_baseline(train_sentences, train_labels, dev_sentences,
                                                                  dev_labels)

    print("Test acuracy for random test:", random_accuracy)
    print("Test acuracy for random dev:", random_accuracy_dev)

    length_accuracy, length_predictions = length_baseline(train_sentences, train_labels, test_input, test_labels)
    length_accuracy_dev, length_predictions_dev = length_baseline(train_sentences, train_labels, dev_sentences, dev_labels)

    print("Test acuracy for length test:", length_accuracy)
    print("Test acuracy for length dev:", length_accuracy_dev)

    frequency_accuracy, frequency_predictions = frequency_baseline(train_sentences, train_labels, test_input,
                                                                   test_labels)
    frequency_accuracy_dev, frequency_predictions_dev = frequency_baseline(train_sentences, train_labels, dev_sentences,
                                                                           dev_labels)

    print("Test acuracy for frequency test:", frequency_accuracy)
    print("Test acuracy for frequency dev:", frequency_accuracy_dev)

    majority_predictions.to_csv("experiments/baselines/majority_test.csv", index=False)
    random_predictions.to_csv("experiments/baselines/random_test.csv", index=False)
    length_predictions.to_csv("experiments/baselines/length_test.csv", index=False)
    frequency_predictions.to_csv("experiments/baselines/frequency_test.csv", index=False)

