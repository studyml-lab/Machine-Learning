# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 14:06:44 2018

@author: venkat
"""
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense

from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
os.chdir("E:\\deep_learning")


from utilities import plot_data, plot_confusion_matrix, plot_loss_accuracy, plot_decision_boundary

iris = datasets.load_iris()




irish_Data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])


irish_Data.info()
irish_Data.shape
labels1 = irish_Data["target"].values.tolist()

data1 =  irish_Data.iloc[:,0:4].as_matrix()

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
    plt.savefig(filename)
    
    


tsne = TSNE(perplexity=30.0, n_components=2, n_iter=5000)
low_dim_embqedding = tsne.fit_transform(data1)

plot_with_labels(low_dim_embqedding, labels1)

target_y  = np_utils.to_categorical(irish_Data["target"])

model = Sequential()
model.add(Dense(units = 64, input_shape=(4,), activation='tanh'))
model.add(Dense(units = 3,  activation='softmax'))
print(model.summary())

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 30
batchsize = 16
history = model.fit(x=data1, y=target_y, verbose=2, epochs=epochs, batch_size=batchsize)
print(model.get_weights())
plot_loss_accuracy(history)
