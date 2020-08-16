# -*- coding: utf-8 -*-
"""
Created on Sat May  2 15:58:32 2020

@author: Krishnav
"""

import numpy as np
import random
import csv
from collections import Counter
import imblearn
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
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

#defining keras model
model= Sequential()
model.add(Dense(20, input_dim=13, kernel_initializer='uniform',activation='relu'))
model.add(Dense(5, activation='softmax'))

#compiling our model

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#data preprocessing

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


print("The data before sampling:\n",Counter(dataset[:,13]))


#classifying the data targets into multiple classes
output=[]
print(len(dataset))
for i in range(len(dataset)):
    if(dataset[i][13]=='0'):
        output.append([1,0,0,0,0])
    if(dataset[i][13]=='1'):
        output.append([0,1,0,0,0])
    if(dataset[i][13]=='2'):
        output.append([0,0,1,0,0])
    if(dataset[i][13]=='3'):        
        output.append([0,0,0,1,0])
    if(dataset[i][13]=='4'):
        output.append([0,0,0,0,1])

print(len(output))
output=np.array(output)
inputs=dataset[:,0:13]
scaler = StandardScaler()
scaler.fit(dataset[:,0:13])
inputs= scaler.transform(inputs)
#overfitting the data
oversample = SMOTE()

train_inputs, test_inputs, train_output, test_output = train_test_split(inputs, output, test_size=0.20)

train_inputs, train_output= oversample.fit_sample(train_inputs, train_output)
  
print(len(train_inputs))
print(len(train_output))
#training the model
print("Training the model")
t0=time.clock()
history= model.fit(train_inputs, train_output, epochs= 500, batch_size= 10)
t1= time.clock()-t0
print("Time for training: ",t1)
performance=model.evaluate(train_inputs, train_output)
print("Average Loss: ",performance[0],", Average Accuracy:",performance[1])

#plotting accuracy
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
#plotting loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


test_output_class=[]
for i in range(len(test_output)):
    k=test_output[i].tolist().index(1)
    test_output_class.append(k)
   
#making predictions
print("\nMaking predictions on the test data")
t0=time.clock()
predictions = model.predict(test_inputs)
t1=time.clock()-t0
print("time for predicting:",t1)
predictions2 = model.predict_classes(test_inputs)

#evaluating the model
performance=model.evaluate(test_inputs, test_output)

error=0
for i in range(len(test_inputs)): 
    print( test_output_class[i]," Predicted output ->", predictions2[i])
    if(test_output_class[i]!=predictions2[i]):
        print("error")
        error=error+1
    
error=error/len(test_inputs)

print("test_report",confusion_matrix(test_output_class , predictions2))
print(classification_report(test_output_class, predictions2))

print("Average Loss: ",performance[0],", Average Accuracy:",performance[1])
print("Error: ",error)
#storing the results
store("ANN.csv",performance[1],error)

