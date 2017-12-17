
import os
import sklearn
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

text_file = open("D:/NLPTask/LabelledData.txt", "r") #Absolute file path
lines = text_file.readlines()
#print(lines)
text_file.close()

df=pd.DataFrame(columns= ['ques', 'label'])
df['ques'].astype('str')

#Read the text line-by-line
for line in lines: 
    #Split the question and label
    ques=line.split(",,,")[0]
    label=line.split(",,,")[1].strip()

    #Convert into numeric label
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

#df.to_csv("D:/NLPTask/Data.csv")
#print(df.shape)

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

'''
y_train=y_train.astype('int')
y_pred_train=[0]*y_train.shape[0]

n=0
for sample in X_train:
    #print(sample)
    sample=np.array2string(sample)
    sample=sample.strip('[\'')
    sample=sample.strip('[ \'')

    sample=sample.strip('["')
    sample=sample.strip('[ "')

    sample=sample.strip('\']')
    sample=sample.strip('"]')

    word=sample.split()[0:1]
    #Unknown    
    y_pred=4
    #When
    if sample.find("when")!=-1:
        y_pred=2
    elif (sample.find("what")!=-1 and sample.find("time")!=-1):
        y_pred=2
    elif (sample.find("which")!=-1 and sample.find("year")!=-1):
        y_pred=2
    #What
    elif sample.find("what")!=-1:
        y_pred=1    
    #Who
    elif sample.find("who")!=-1:
        y_pred=0
    #Affirmation
    elif (word=="is" or word=="will" or word=="does" or word=="do" or word=="can" or word=="are" or word=="has" or word=="could"):
        y_pred=3
    
    #elif (word=="how" or word=="name" or word=="why" or word=="where" or word="whose" or word="which"):
     #   y_pred=4
    y_pred_train[n]=y_pred
    #print(sample)
    #print(y_train[n])
    n=n+1
    #print(y_pred)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_train, y_pred_train))
#ans=0.893063583815
'''
#Test data
y_test=y_test.astype('int')
y_pred_test=[0]*y_test.shape[0]
n=0
for sample in X_test:
    #print(sample)
    sample=np.array2string(sample)
    sample=sample.strip('[\'')
    sample=sample.strip('[ \'')

    sample=sample.strip('["')
    sample=sample.strip('[ "')

    sample=sample.strip('\']')
    sample=sample.strip('"]')

    word=sample.split()[0:1]
    #Unknown    
    y_pred=4
    #When
    if sample.find("when")!=-1:
        y_pred=2
    elif (sample.find("what")!=-1 and sample.find("time")!=-1):
        y_pred=2
    elif (sample.find("which")!=-1 and sample.find("year")!=-1):
        y_pred=2
    #What
    elif sample.find("what")!=-1:
        y_pred=1    
    #Who
    elif sample.find("who")!=-1:
        y_pred=0
    #Affirmation
    elif (word=="is" or word=="will" or word=="does" or word=="do" or word=="can" or word=="are" or word=="has" or word=="could"):
        y_pred=3
    y_pred_test[n]=y_pred

    n=n+1
    
print(accuracy_score(y_test, y_pred_test))
#ans=0.874157303371