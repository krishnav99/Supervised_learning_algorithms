# -*- coding: utf-8 -*-
"""
Created on Tue May 12 23:56:48 2020

@author: Krishnav
"""


# -*- coding: utf-8 -*-
"""
Created on Sat May  2 15:58:32 2020

@author: Krishnav
"""

import numpy as np
import random
import csv
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import time
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from sotrer import store

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


#reading the database
dataset=[]
with open('heart.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        dataset.append(row)

dataset.pop(0)



#defining keras model
model= Sequential()
model.add(Dense(20,input_dim=13,kernel_initializer='uniform',activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#compiling our model
model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])

#data preprocessing
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

train_inputs, train_output= oversample.fit_sample(train_inputs, train_output)

print(len(train_inputs))
print(len(train_output))

#training the model
t0 = time.clock()
print("Training the model")
history= model.fit(train_inputs, train_output, epochs= 500, batch_size= 10)
t1= time.clock()-t0
print("Time for training: ",t1)

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


#making predictions
print("\nMaking predictions on the test data")

t0 = time.clock()
predictions = model.predict_classes(test_inputs)
t1= time.clock()-t0
print("Time for prediction: ",t1)

error=0
for i in range(len(test_inputs)): 
    print( test_output[i]," Predicted output ->", predictions[i])
    if(int(test_output[i])!=predictions[i]):
        print("error")
        error=error+1
        
#evaluating the model
performance=model.evaluate(test_inputs, test_output)
print("Average Loss: ",performance[0],", Average Accuracy:",performance[1])
print("error:", error/i)
test_output= test_output.astype(np.int) 
print("test_report\n",confusion_matrix(test_output , predictions))
print(classification_report(test_output, predictions))

#storing the results
store("ANN.csv",performance[1],error)