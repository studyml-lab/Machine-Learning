# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 16:12:33 2018

@author: venkat
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
os.getcwd()
os.chdir("E:\\deep_learning")
from utilities import plot_data, plot_confusion_matrix, plot_loss_accuracy, plot_decision_boundary
from sklearn.datasets import make_classification
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
import pydot
import graphviz
import pandas as pd
import numpy as np
from keras.utils import np_utils
pydot.find_graphviz = lambda: True

input = pd.read_csv("sample1.csv")



X_train = input.iloc[:,1:3].as_matrix()
y_train = input["Output"].as_matrix()

model1 = Sequential()
model1.add(Dense(units =1,input_shape=(2,),activation='sigmoid'))

#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model1.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
history1 = model1.fit(x = X_train,y = y_train,verbose=0,epochs=1)



print(model1.get_weights())
plot_loss_accuracy(history1)
plot_decision_boundary(lambda X_train: model1.predict(X_train),X_train,y_train)

y_pred = model1.predict_classes(X_train, verbose=0)

plot_confusion_matrix(model1,X_train,y_train)