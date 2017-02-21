from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.models import Sequential
import os
import csv
import scipy.io
import numpy as np
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import loader

#------------------------------------------------------------------------------
# Read Matlab .mat file
#------------------------------------------------------------------------------
def readMat(fmat):  
  ds = scipy.io.loadmat(fmat)
  labels = [x for x in ds.keys() if x[0] != '_']
  label = labels[0]
  nsamples = ds[label].shape[0]

  #print fmat+" contains: "+str(ds[label].shape[0])+" samples with "+str(ds[label].shape[1])+" features"
  return ds[label]#, ds[label].shape[1]
#------------------------------------------------------------------------------


def convolution(signal, window_size, step, fadein=0, fadeout=0):
    start = fadein
    output = []
    while start + window_size < len(signal):
        output.append(signal[start:start+window_size])
        start+=step
        
    return output

def plot_signal(signal, fig=plt.figure()):
    plt.clf()
    time = np.arange(0, len(signal))
    plt.scatter(time, signal, s = 1) # 
    
              
# Creates a directory 
def create_training_set(X, Y, window_size, step, fadein=0, fadeout=0):
    
    for x in X:
        out = convolution(M[0], 100, 10)
    
    
X, Y = loader.load_all_data
create_training_set

    


"""
# this creates a model that includes
# the Input layer and three Dense layers
model = Sequential()
model.add(Dropout(0.1, input_shape=(MAX_FEAT, )))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='adagrad',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
              
print M.shape
y_binary = to_categorical(labels)
model.fit(M, y_binary, validation_split=0.33)  # starts training
"""
