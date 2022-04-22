# The original code only outputs the accuracy and the loss.
# Process the file model_output_00005.tsv and calculate precision, recall, and F1 for each class

""" For part C, we use an implementation for a vanilla LSTM which was originally developed for a
named entity recognition project for a Stanford course.
"""

from sklearn.metrics import classification_report
import pandas as pd

# 12. Detailed evaluation (2.5 points)

# Train the model on the data in preprocessed/train and preprocessed/dev by running the
# code in train.py.
# Evaluate the model on the data in preprocessed/test by running evaluate.py.
# The original code only outputs the accuracy and the loss of the model. I adapted the
# code, so that it writes the predictions to experiments/base_model/model_output_00005.tsv.

# Implement calculations for precision, recall, and F1 for each class in
# TODO_detailed_evaluation.py. You can use existing functions but make sure that you
# understand how they work. Provide the results for the baselines and the LSTM in the table below.

base_model = pd.read_csv("experiments/base_model/model_output.tsv", sep='\t', header=None)
base_model.columns = ['tokens', 'gold', 'predictions']
frequency_model = pd.read_csv("experiments/baselines/frequency_test.csv")
length_model = pd.read_csv("experiments/baselines/length_test.csv")
majority_model = pd.read_csv("experiments/baselines/majority_test.csv")
random_model = pd.read_csv("experiments/baselines/random_test.csv")

final_df = pd.concat([base_model, frequency_model[['tokens', 'predictions']]], axis=1)
final_df = pd.concat([final_df, length_model[['predictions']]], axis=1)
final_df = pd.concat([final_df, majority_model[['predictions']]], axis=1)
final_df = pd.concat([final_df, random_model[['predictions']]], axis=1)

final_df.columns = ['tokens', 'gold', 'base_model', 'my_tokens', 'frequency_model', 'length_model', 'majority_model',
                    'random_model']

for i in range(len(final_df)):
    current_token = final_df['my_tokens'].iloc[i]
    if "\n" in str(current_token):
        final_df.iloc[i + 1:, 3:] = final_df.iloc[i + 1:, 3:].shift(periods=1)
        # print("STOP")

print("Loading data .. ")


def get_report(model_name, y_true, y_pred):
    # mapping -> 1: N, 0: C

    def convert_name(label):
        if label == "N":
            label = 1
        else:
            label = 0
        return label

    y_true_list = list(y_true)
    y_pred_list = list(y_pred)

    y_true_list = list(map(convert_name, y_true_list))
    y_pred_list = list(map(convert_name, y_pred_list))

    print(model_name, " classification report")
    print(classification_report(y_true_list, y_pred_list, labels=[1, 0]))


model_names = ['base_model', 'frequency_model', "length_model", "majority_model", "random_model"]

for name in model_names:
    get_report(name, final_df['gold'], final_df[name])
