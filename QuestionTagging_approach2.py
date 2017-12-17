import os
import csv
import pandas as pd
import sklearn
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

text_file = open("D:/NLPTask/LabelledData.txt", "r") 
lines = text_file.readlines()
#print(lines)
text_file.close()


df=pd.DataFrame(columns= ['ques', 'label'])
df['ques'].astype('str')


for line in lines: 
    ques=line.split(",,,")[0]
    label=line.split(",,,")[1].strip()
    # Who=0, What=1, When=2, Affirmation(yes/no)=3,  Unknown=4 
    if label=="who":
        label=0
    elif label=="what":
        label=1
    elif label=="when":
        label=2
    elif label=="affirmation":
        label=3
    elif label=="unknown" :
        label=4
    
    #print(ques)
    #print(label)
    
    row = [ques,label]
    df.loc[len(df)] = row;

data=df

train=data.drop('label',axis=1);
target=data['label']

X=train.values;
y=target.values;
sss = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
sss.get_n_splits(X, y);

for train_index, test_index in sss.split(X, y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

#print(X_train)
#print(y_train)
y_train=y_train.astype('int')
y_test=y_test.astype('int')

#NB
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train.ravel())
#X_train_counts.shape=(1380,2790)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = BernoulliNB().fit(X_train_tfidf, y_train)
X_new_counts = count_vect.transform(X_test.ravel())
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

y_pred_test = clf.predict(X_new_tfidf)

print(np.mean(y_pred_test == y_test))
    #0.797752808989

y_pred_test=y_pred_test.astype('int')

metrics.confusion_matrix(y_test, y_pred_test)

#SVM
#count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform(X_train.ravel())
#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf=SGDClassifier().fit(X_train_tfidf, y_train)
X_new_counts = count_vect.transform(X_test.ravel())
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

y_pred_test = clf.predict(X_new_tfidf)

print(np.mean(y_pred_test == y_test))
#ans=0.957303370787

y_pred_test=y_pred_test.astype('int')

metrics.confusion_matrix(y_test, y_pred_test)