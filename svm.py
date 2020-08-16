# -*- coding: utf-8 -*-
"""
Created on Sat May  2 15:58:32 2020

@author: Krishnav
"""

import numpy as np
import csv
import random 
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import sklearn.metrics as sk
from sotrer import store
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import time
from sklearn import svm

#reading the database
dataset=[]
length=[]

with open('heart.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        dataset.append(row)
dataset.pop(0)

#dataprocessing
dataset=np.array(dataset)
random.shuffle(dataset)  
output=dataset[:,13]
inputs=dataset[:,0:13]
scaler = StandardScaler()
scaler.fit(dataset[:,0:13])
inputs= scaler.transform(inputs)

oversample = SMOTE()

train_inputs, test_inputs, train_output, test_output = train_test_split(inputs, output, test_size=0.20)

train_inputs, train_output= oversample.fit_sample(train_inputs, train_output)

print(len(train_inputs))
print(len(train_output))
#training the algorithm
classifier=svm.SVC(kernel='linear')
t0=time.clock()
classifier.fit(train_inputs, train_output)
t1=time.clock()-t0
print("time for training = ",t1)
t0=time.clock()
prediction=classifier.predict(test_inputs)
t1= time.clock()-t0
print("time for prediction:",t1)
#evaluating the algorithm
accuracy = sk.accuracy_score(test_output, prediction)

print("Predicting using knn")

for i in range(len(test_inputs)):
    print(test_output[i]," Predicted output ->", prediction[i][0])
    if(test_output[i]!=prediction[i][0]):
        print("error")
        error=(np.mean(prediction!=test_output))    

#evaluating model
print()
print("Accuracy:", accuracy," Error:",error)
print("test_report\n",confusion_matrix(test_output , prediction))
print(classification_report(test_output, prediction))


#calculating error


#storing the data for future reference
#store("kNN.csv",accuracy,error) 