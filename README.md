# NLP-categorize-the-questions
Given a question, the aim is to identify the category ("who", "what", "when", "affirmation" or "unknnown" ) it belongs to. 

Example:
1. What is your name? Type: What
2. When is the show happening? Type: When
3. Is there a cab available for airport? Type: Affirmation
There are ambiguous cases to handle as well like:
What time does the train leave(this looks like a what question but is actually a When type)

DESCRIPTION OF DATASET:
how did serfdom develop in and then leave russia ? ,,, unknown
what films featured the character popeye doyle ? ,,, what
...
can i get it in india ? ,,, affirmation
would this work on a 2008 ford edge with a naked roof ? ,,, affirmation

Total: 1483 questions/samples

LANGUAGE USED: 
Python (Here, version 3.6.3)

PACKAGES REQUIRED:
numpy, os, sklearn, pandas and associated dependencies.

To install and work with Python, install Anaconda software (a cross-platform Python distribution with most of the required packages and dependencies  preinstalled). Refer: https://docs.anaconda.com/anaconda/install/

PREPROCESSING OF DATA:
To work with the given text data in Python, Dataframe structure of pandas library is used. Each of the sample is split into "question" and "label". Thus, the dimension of datastructure is 1483x2. For ease, the labels are converted to numeric datatype (Who=0, What=1, When=2, Affirmation(yes/no)=3,  Unknown=4).

Split the dataset into training and testing datasets. "Stratified Split"  is used for this purpose. Refer: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html

Here, training dataset has 1038 samples and test dataset has 445 samples (70%-30%)

APPROACH I:
Form the manual observation of training samples, to correctly tag the question, look for the "keywords" in the question sentence. Affirmation type of questions begin with the forms of verb "be". "when" type of questions mostly begin with "when" or consist of "what time"/"whcih year" phrases. Except the affirmation type of questions, the placement of the keyword is not fixed. Default type is assigned as unknown. So, samples not classified in any of the four categories are by default tagged as "unknown" type.

Percentage of correctly classified samples: Accuracy score=0.874157303371

APPROACH II:

tf–idf (term frequency–inverse document frequency)
Convert the text context into numerical feature vectors. A fixed integer id is assigned to each word occuring in the training dataset and for each document (Here, each sample), the freuqncy of a word occuring in that document is stored (scikit-learn stores these as sparse feature vetors, i.e., storing only the non-zero frquency terms). The weight of term is proportional  to the number of times it occurs in a document. "inverse document frequency" factor is incorporated which diminishes the weight of terms that occur very frequently in the document set and increases the weight of terms that occur rarely.

Training the classifier
Models used are Naive Bayesian classifer and Support Vector Machine.

The accuracy of the models are:
NB= 0.752808988764
SVM=0.957303370787
Confusion matrix is also another metric used to evaluate the performance of the model





