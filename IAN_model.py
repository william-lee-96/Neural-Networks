#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 14:15:24 2022

@author: williamlee
"""

import pandas as pd
import numpy as np
import os

import glob

###############################################################################

from sklearn.preprocessing import MinMaxScaler

def extract_normalized_data(sequences, n_steps):
    X, y = [], []
    for i in range(len(sequences)):
        end_idx = i + n_steps + 1
        if end_idx > len(sequences):
            break
        scaler = MinMaxScaler()
        data = sequences[i:end_idx, :]
        scaler.fit(data)
        norm_data = scaler.transform(data)
        seq_x, seq_y = norm_data[:-1], norm_data[-1]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)
    
###############################################################################

n_steps = 10

X_list = []
y_list = []

# data downloaded on 03/15/2022
for file in os.scandir('/kaggle/input/ian-data/'):
    
    company_df = pd.read_csv(file.path)
    
    company_data = company_df.iloc[:,1:].values
        
    X, y = extract_normalized_data(company_data, n_steps)
    
    X_list.append(X)
    y_list.append(y)
    
X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)   

###############################################################################

X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))

print(X.shape)
print(y.shape)

n_output = y.shape[1]
n_features = X.shape[2]

###############################################################################

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import History

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=1)
history = History()

###############################################################################

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D

from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional

from tensorflow.keras.optimizers import Adam

from tensorflow.distribute.cluster_resolver import TPUClusterResolver
from tensorflow.distribute.experimental import TPUStrategy

tpu = TPUClusterResolver.connect()
tpu_strategy = TPUStrategy(tpu)

with tpu_strategy.scope():
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=128, kernel_size=9, activation='relu', padding='causal', input_shape=(n_steps, n_features))))
    model.add(TimeDistributed(Conv1D(filters=256, kernel_size=7, activation='relu', padding='causal')))
    model.add(TimeDistributed(Conv1D(filters=512, kernel_size=5, activation='relu', padding='causal')))
    model.add(TimeDistributed(Conv1D(filters=1024, kernel_size=3, activation='relu', padding='causal')))
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(100, activation='relu', return_sequences=True)))
    model.add(Bidirectional(LSTM(100, activation='relu', return_sequences=True)))
    model.add(Bidirectional(LSTM(100, activation='relu')))
    model.add(Dense(100))
    model.add(Dense(n_output))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
# fit model
model.fit(X, y, epochs=1000, verbose=1, callbacks=[es, history], validation_split=0.2, batch_size=2048)

###############################################################################

val_losses = history.history['val_loss']

num_epochs = len(val_losses)
best_val_loss = min(val_losses)
best_epoch = val_losses.index(best_val_loss) + 1

print(f'num_epochs: {num_epochs} | best_epoch: {best_epoch} | best_val_loss: {best_val_loss}')