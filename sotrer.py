# -*- coding: utf-8 -*-
"""
Created on Sun May 10 12:38:18 2020

@author: Krishnav
"""
import matplotlib.pyplot as plt
import csv
import numpy as np

length=[]
def store(csvfile, accuracy, loss):
    flag=0
    file = open(csvfile, 'a+') #creates the file
    file.close()
    file = open(csvfile, 'r') #reads the no. of rows of the file
    with file:
        reader = csv.reader(file)
        length=list(reader)
        if(len(length)>=5):  
            flag = 1
        print("test no: ",len(length)+1)        
    file.close()
    if(flag==1):
        file=open(csvfile, 'w') #erase the file if row = 5 
        file.close()
    file=open(csvfile,'a',newline='') #adds a line in the file 
    with file:
        writer=csv.writer(file)
        writer.writerow([accuracy,loss])
        
def observe(csvfile):   
    accuracy=[]
    loss=[]
    with open(csvfile) as file:
        reader  = csv.reader(file)
        for row in reader:
            accuracy.append(row[0])
            loss.append(row[1])
    #accuracy = np.array(accuracy)
    #loss = np.array(loss)
    return accuracy, loss

knn_observation = observe("kNN.csv")
Ann_observation = observe("ANN.csv")

def calculate():
    for i in range(0,5):
        knn_observation[0][i]=float(knn_observation[0][i])
        knn_observation[1][i]=float(knn_observation[1][i])

    for i in range(0,5):
        Ann_observation[0][i]=float(Ann_observation[0][i])
        Ann_observation[1][i]=float(Ann_observation[1][i])
    print("kNN")
    print("Average Accuracy:",np.mean(knn_observation[0]),"Average loss:",
                                      np.mean(knn_observation[0]))
    print("Backprpagation")
    print("Average Accuracy:",np.mean(Ann_observation[0]),"Average loss:",
                                      np.mean(Ann_observation[0]))
    
