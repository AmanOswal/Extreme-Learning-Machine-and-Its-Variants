#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#size of identity matrix is too large to run on a normal computer
import pandas as pd
import numpy as np
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#Reshaping into 2-D dataframe

x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

x_train = x_train[0:10000,:]
y_train = y_train[0:10000]
x_train = pd.DataFrame(x_train)
y_train = pd.DataFrame(y_train)
#ELM
w = np.random.rand(784,6)#784,6
b = np.random.rand(1,6)#1,6
h = np.dot(x_train,w)+b#60000,6
beta=np.dot(np.dot(h.T, np.linalg.inv(np.dot(h,h.T) + np.identity(60000))),y_train)
res = np.dot(h,beta)
#Evaluating
res =np.round(res)#rounding off to nearest integer
from sklearn.metrics import accuracy_score
score = accuracy_score(y_train,res)
print(score)

