# Implement four baselines for the task.
# Majority baseline: always assigns the majority class of the training data
# Random baseline: randomly assigns one of the classes. Make sure to set a random seed and average the accuracy over 100 runs.
# Length baseline: determines the class based on a length threshold
# Frequency baseline: determines the class based on a frequency threshold

from model.data_loader import DataLoader

# Each baseline returns predictions for the test data. The length and frequency baselines determine a threshold using the development data.

def majority_baseline(train_sentences, train_labels, testinput, testlabels):
    predictions = []

    # TODO: determine the majority class based on the training data
    # ...
    majority_class = "X"
    predictions = []
    for instance in testinput:
        tokens = instance.split(" ")
        instance_predictions = [majority_class for t in tokens]
        predictions.append(instance, instance_predictions)

    # TODO: calculate accuracy for the test input
    # ...
    accuracy = ''
    return accuracy, predictions


def random_baseline(train_sentences, train_labels, testinput, testlabels):
    predictions = []

    pass


def length_baseline(train_sentences, train_labels, testinput, testlabels):
    predictions = []

    pass


def frequency_baseline(train_sentences, train_labels, testinput, testlabels):
    predictions = []

    pass



if __name__ == '__main__':
    train_path = "data/preprocessed/train"
    dev_path = "data/preprocessed/dev"
    test_path = "data/preprocessed/test"

    # Note: this loads all instances into memory. If you work with bigger files in the future, use an iterator instead.

    with open(train_path + "sentences.txt") as sent_file:
        train_sentences = sent_file.readlines()

    with open(train_path + "labels.txt") as label_file:
        train_labels = label_file.readlines()

    with open(dev_path + "sentences.txt") as dev_file:
        dev_sentences = dev_file.readlines()

    with open(train_path + "labels.txt") as dev_label_file:
        dev_labels = dev_label_file.readlines()

    with open(test_path + "sentences.txt") as testfile:
        test_input = testfile.readlines()

    with open("test_path + labels.txt") as test_label_file:
        test_labels = test_label_file.readlines()

    majority_accuracy, majority_predictions = majority_baseline(train_sentences, train_labels, test_input, test_labels)
    majority_accuracy_dev, majority_predictions_dev = majority_baseline(train_sentences, train_labels, dev_sentences, dev_labels)

    print("Test acuracy for majority test:", majority_accuracy)
    print("Test acuracy for majority dev:", majority_accuracy_dev)

    random_accuracy, random_predictions = random_baseline(train_sentences, train_labels, test_input, test_labels)
    random_accuracy_dev, random_predictions_dev = random_baseline(train_sentences, train_labels, dev_sentences, dev_labels)

    print("Test acuracy for random test:", random_accuracy)
    print("Test acuracy for random dev:", random_accuracy_dev)

    length_accuracy, length_predictions = length_baseline(train_sentences, train_labels, test_input, test_labels)
    length_accuracy_dev, length_predictions_dev = length_baseline(train_sentences, train_labels,  dev_sentences, dev_labels)

    print("Test acuracy for length test:", length_accuracy)
    print("Test acuracy for length dev:", length_accuracy_dev)

    frequency_accuracy, frequency_predictions = frequency_baseline(train_sentences, train_labels, test_input, test_labels)
    frequency_accuracy_dev, frequency_predictions_dev = frequency_baseline(train_sentences, train_labels, dev_sentences, dev_labels)

    print("Test acuracy for frequency test:", frequency_accuracy)
    print("Test acuracy for frequency dev:", frequency_accuracy_dev)

    # TODO: output the predictions in a suitable way so that you can evaluate them

    # majority_predictions.to_csv()