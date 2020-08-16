# -*- coding: utf-8 -*-
"""
Created on Mon May 18 21:34:21 2020

@author: Krishnav
"""
import numpy as np
import random
import csv
from collections import Counter
import imblearn
import sklearn.metrics as sk
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sotrer import store
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
import time

#reading the database
dataset=[]
with open('processed.cleveland.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        dataset.append(row)



#removing the noisy data
index=[]
for i in range(len(dataset)):
    for k in range(len(dataset[1])):
        if(dataset[i][k]=='?'):
            print(i," ",k)
            index.append(i)
                
print(index)
for i in range(len(index)):
    dataset.pop(index[i]-i)
   
#shuffeling the data
dataset=np.array(dataset)
random.shuffle(dataset)  
output=dataset[:,13]
inputs=dataset[:,0:13]
scaler = StandardScaler()
scaler.fit(dataset[:,0:13])
inputs= scaler.transform(inputs)
#overfitting the data
oversample = SMOTE()

train_inputs, test_inputs, train_output, test_output = train_test_split(inputs, output, test_size=0.20)
print("The data before sampling:\n",Counter(train_output))

train_inputs, train_output= oversample.fit_sample(train_inputs, train_output)
  
print("The data after sampling:\n",Counter(train_output))

print(len(train_inputs))
print(len(train_output))
#training the algorithm
classifier=KNeighborsClassifier(n_neighbors=7)
classifier.fit(train_inputs, train_output)

t0=time.clock()
prediction=classifier.predict(test_inputs)
t1=time.clock()-t0
print("time for prediction ", t1)
#evaluating the algorithm
accuracy = sk.accuracy_score(test_output, prediction)

print("Predicting using knn")

for i in range(len(test_inputs)):
    print(test_output[i]," Predicted output ->", prediction[i][0])
    if(test_output[i]!=prediction[i][0]):
        print("error")
        
error=(np.mean(prediction!=test_output))    

print("Accuracy:", accuracy," Error:",error)
print("test_report\n",confusion_matrix(test_output , prediction))
print(classification_report(test_output, prediction))
#calculating error

errorlist=[]
for i in range(1, 40):
    if(i%2!=0 or i==2):
            classifier = KNeighborsClassifier(n_neighbors=i)
            classifier.fit(train_inputs, train_output)
            prediction=classifier.predict(test_inputs)
            errorlist.append(np.mean(prediction!=test_output))
plt.plot(errorlist,marker='o')
plt.show()


#storing the data for future reference


