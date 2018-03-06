# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 09:03:13 2018

@author: venkat
"""

from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from keras.utils import np_utils
import os
os.chdir("E:\\deep_learning")
from utilities import plot_data, plot_confusion_matrix, plot_loss_accuracy, plot_decision_boundary
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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
    
np.random.seed(100)

digit_train = pd.read_csv("train.csv")
digit_train.shape
digit_train.info()



data1 =  digit_train.iloc[1:1000,1:].as_matrix()
labels1 = digit_train["label"].values.tolist()
data1.shape

tsne = TSNE(perplexity=30.0, n_components=2, n_iter=5000)
low_dim_embqedding = tsne.fit_transform(data1)

plot_with_labels(low_dim_embqedding, labels1)

X_train = digit_train.iloc[:,1:].as_matrix()
y_train = np_utils.to_categorical(digit_train["label"])


model = Sequential()
model.add(Dense(units = 64, input_shape=(784,), activation='tanh'))
model.add(Dense(units = 32, activation='tanh'))
model.add(Dense(units = 16, activation='tanh'))

model.add(Dense(units = 10,  activation='softmax'))
print(model.summary())

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 30
batchsize = 16
history = model.fit(x=X_train, y=y_train, verbose=2, epochs=epochs, batch_size=batchsize, validation_split=0.2)

#model2 ...................

model2 = Sequential()
model2.add(Dense(units = 248, input_shape=(784,), activation='tanh'))
model2.add(Dense(units = 124, activation='tanh'))
model2.add(Dense(units = 62, activation='tanh'))
model2.add(Dense(units = 32, activation='tanh'))
model2.add(Dense(units = 16, activation='tanh'))
model2.add(Dense(units = 10,  activation='softmax'))
print(model2.summary())

model2.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 35
batchsize = 16
history2 = model2.fit(x=X_train, y=y_train, verbose=2, epochs=epochs, batch_size=batchsize, validation_split=0.2)





print(model.get_weights())
plot_loss_accuracy(history2)

digit_test = pd.read_csv("test.csv")
digit_test.shape
digit_test.info()

X_test = digit_test.as_matrix()
pred = model2.predict_classes(X_test)
submissions=pd.DataFrame({"ImageId": list(range(1,len(pred)+1)),
                         "Label": pred})
submissions.to_csv("submission34_v2.csv", index=False, header=True)
