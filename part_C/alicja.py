"""
For part C, we use an implementation for a vanilla LSTM which was originally developed for a
named entity recognition project for a Stanford course. You can find more documentation here:
https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/nlp


11. Understanding the code (1.5 Points)
Familiarize yourself with our version of the code and try to understand what is going on.
Answer in your own words (1-3 sentences per question)
Run the file build_vocab.py. What does this script do?
Inspect the file model/net.py. Which layers are being used and what is their function?
How could you change the loss function of the model?
"""

# TODO: what "our" version of the code? where is it



"""
Detailed evaluation (2.5 points)
Train the model on the data in preprocessed/train and preprocessed/dev by running the 
code in train.py. 
Evaluate the model on the data in preprocessed/test by running evaluate.py. 
The original code only outputs the accuracy and the loss of the model. I adapted the 
code, so that it writes the predictions to experiments/base_model/model_output.tsv.
Implement calculations for precision, recall, and F1 for each class in 
TODO_detailed_evaluation.py. You can use existing functions but make sure that you 
understand how they work. 
Provide the results for the baselines and the LSTM in the table below.
"""




"""
Interpretation (1.5 Points)
Compare the performance to the results in the shared task 
(https://aclanthology.org/W18-0507.pdf) and interpret the results in 3-5 sentences. Don’t 
forget to check the number of instances in the training and test data and integrate this 
into your reflection.  
"""



"""
Experiments (2 points)
Vary a hyperparameter of your choice and plot the F1-results (weighted average) for at 
least 5 different values. Examples for hyperparameters are embedding size, learning 
rate, number of epochs, random seed,
Hyperparameter: 
Plot:
Interpret the result (2-4 sentences):


Provide 3 examples for which the label changes when the hyperparameter changes:
1. Example 1, Label at Value 1, Label at Value 2
2. Example 2, Label at Value 1, Label at Value 2
3. Example 3, Label at Value 1, Label at Value 2
"""



