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
pydot.find_graphviz = lambda: True


x,y = make_classification(n_samples=100,n_informative=2,n_features=2,n_redundant=0,
                          n_clusters_per_class=1,random_state=7)

y

model = Sequential()
model.add(Dense(units =1,input_shape=(2,),activation='sigmoid'))

#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x = x,y = y,verbose=0,epochs=200)

print(model.get_weights())
print(model.summary())
plot_loss_accuracy(history)
plot_decision_boundary(lambda x: model.predict(x),x,y)

y_pred = model.predict_classes(x, verbose=0)

plot_confusion_matrix(model,x,y)