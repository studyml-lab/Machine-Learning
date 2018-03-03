# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:30:39 2017

@author: venkat
"""

import os
os.getcwd()
os.chdir("E:\\deep_learning")
from utilities import plot_data, plot_confusion_matrix, plot_loss_accuracy, plot_decision_boundary
from sklearn.datasets import make_moons,make_circles
from keras.models import Sequential
from keras.layers import Dense
 
x,y = make_circles(n_samples=1000,noise = 0.05,random_state=0,factor=0.3)

plot_data(x,y)

model = Sequential()
model.add(Dense(units=4,activation='tanh',input_shape=(2,)))
model.add(Dense(units=2,activation='tanh'))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(x,y,epochs=1,verbose=0)
plot_decision_boundary(lambda x: model.predict(x),x,y)
plot_confusion_matrix(model,x,y)