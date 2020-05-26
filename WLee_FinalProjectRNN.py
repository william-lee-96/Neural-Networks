#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:33:58 2020
@author: williamlee
"""

import os
import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
from torchtext import data
import random as rand
import torch.optim as optim
from sklearn import metrics

# pass the following commands to the terminal:
# pip install sentencepiece
# conda install -c conda-forge spacy
# python -m spacy download en

###############################################################################

"""
The design of this RNN borrows concepts presented in Aravind Pai's
"Build Your First Text Classification model using PyTorch" [tutorial] (1),
Jibin Mathew's "PyTorch Artificial Intelligence Fundamentals: A recipe-based
approach to design, build and deploy your own AI models with PyTorch 1.x"[E-book] (2),
and PyTorch's "TEXT CLASSIFICATION WITH TORCHTEXT" [official tutorial] (3).

(1) https://www.analyticsvidhya.com/blog/2020/01/first-text-classification-in-pytorch/
(2) https://books.google.com/books/about/PyTorch_Artificial_Intelligence_Fundamen.html?id=2crTDwAAQBAJ
(3) https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

Much credit and gratitude to Mr. Pai, Mr. Mathew, & the good people at PyTorch.
"""

# sets the path
path = "/Users/williamlee/Desktop/machine_learning_code/FinalProject"
os.chdir(path)

# manually seeds RNGs (for reproducibility)
torch.manual_seed(1)
np.random.seed(1)

# loads training set as pandas dataframe 
train_df = pd.read_csv("train.csv")

# "temp" is the "text" column, but with all hyperlinks removed
temp = []
for element in train_df["text"]:
    resultant = re.sub(r'http\S+', '', element)
    temp.append(resultant)

# replace original "text" column with "temp"
train_df["text"] = temp

# save "train_df" as "train_df_fin" (to keep copy of original dataset)
train_df_fin = train_df

# save "train_df_fin" as a csv
train_df_fin.to_csv("train_fin.csv",index=False)

# use "SpaCy" tokenizer for text, lowercase text
txt = data.Field(tokenize="spacy", lower=True, batch_first=True, include_lengths=True)
lab = data.LabelField(dtype = torch.float, batch_first=True)

# "flds" is a list of tuples, each containing a "train_df_fin" column name & 
# its respective field object. First three columns are ignored
flds = [(None, None),(None, None),(None, None),("text",txt),("target", lab)]

# create tabular dataset, store in "trn_dat"
trn_dat = data.TabularDataset(path='train_fin.csv', format="csv", fields=flds, skip_header=True)

# randomly splits training data into "train" (80%) and "valid" (20%)
train,valid = trn_dat.split(split_ratio=0.8,random_state=rand.seed(1))

# builds vocabulary using pre-trained word embeddings (glove.6B.300d) 
txt.build_vocab(train, vectors="glove.6B.300d")
lab.build_vocab(train)

# use BucketIterator for splits (minimizes padding)
train_iterator, valid_iterator = data.BucketIterator.splits(
    (train, valid), 
    batch_size=256,
    sort_key = lambda x:len(x.text),
    sort_within_batch=True,
    device = "cpu")

###############################################################################

# this RNN utilizes bidirectional LSTM
class text_classifier(nn.Module):
    
    # all of the model's layers are declared here
    def __init__(self, num_embed, dim_embed, num_lay, num_hidNode, drop_prob):
        
        super().__init__()
        
        # activation function (best choice: sigmoid)
        self.act = nn.Sigmoid()
    
        # "fully connected" linear (dense) layer; 1 output node
        self.fc = nn.Linear(num_hidNode*2, 1)
        
        # LSTM layer 
        self.lstm = nn.LSTM(input_size = dim_embed,
                            hidden_size = num_hidNode,
                            num_layers = num_lay,
                            batch_first = True,
                            dropout = drop_prob,
                            bidirectional = True)
        
        # embedding layer (dictionary mapping indices -> vectors)
        self.embedding = nn.Embedding(num_embed, dim_embed)

    # defines how the model is run - i.e., how the inputs are passed forward
    def forward(self, text, len_text):
        
        # receives input
        embed = self.embedding(text)
        
        # packs the input Tensor (sequences are already padded)
        packed_seq = nn.utils.rnn.pack_padded_sequence(embed, len_text, batch_first=True)
        
        # the 3 returns are output packed seq, hidden st. tensor, & cell st. tensor, respectively
        packed_out, (hid_st, cell_st) = self.lstm(packed_seq)
        
        # concatenates the hidden state (last forward & backward states) over one dimension
        hid_st = torch.cat((hid_st[-2,:,:], hid_st[-1,:,:]), dim = 1)
        
        # updates input to fully connected layer
        dense_out = self.fc(hid_st)
        
        # determines final output using sigmoid activation function
        fin_out = self.act(dense_out)
        
        return fin_out
    
###############################################################################

# initialize the model
model = text_classifier(num_embed=len(txt.vocab), dim_embed=300,
                        num_lay=2, num_hidNode=32, drop_prob=0.7)

# copy pretrained GloVe embeddings onto model's embedding layer weights matrix
glove_vectors = txt.vocab.vectors
model.embedding.weight.data.copy_(glove_vectors)

# Adam is the best optimizer choice here
optimizer = optim.Adam(model.parameters())

# use BCELoss when Sigmoid is activation function
get_loss = nn.BCELoss()

###############################################################################

def trainer(model, iterator, optimizer, loss_calculator):
    
    # tell model it is training mode 
    model.train()  
    
    # for each new epoch, these will be reset to 0
    loss_per_epoch = 0
    accuracy_epoch = 0
    
    # initialize two empty lists (to calculate AUC score later)
    epoch_guesses = []
    epoch_targets = []
    
    for batch in iterator:
        
        # for each new batch, all gradients will be reset to 0
        optimizer.zero_grad()
        
        # retrieve batch's text examples (padded) and their lengths, respectively
        text_eg, len_txt = batch.text
        
        # get model's predictions
        guess = model(text_eg, len_txt)
        
        # make tensor containing predictions one-dimensional 
        guess = guess.squeeze()

        # calculate the batch loss using BCELoss
        batch_loss = loss_calculator(guess, batch.target)
        
        # round predictions to either 1 or 0 
        binary_guess = torch.round(guess)
        
        # correct predictions match the actual targets
        correctGuess = (binary_guess == batch.target).float() 
        
        # append guesses and targets to their respective lists (initialized before loop)
        epoch_guesses.extend(binary_guess.detach().numpy())
        epoch_targets.extend(batch.target.detach().numpy())
            
        # calculate the batch accuracy
        batch_accu = correctGuess.sum()/len(correctGuess)
        
        # using backprop, compute d(loss)/dx for each parameter
        batch_loss.backward()       
        
        # parameters are updated (gradient descent step)
        optimizer.step()      
        
        # update epoch loss & accuracy 
        loss_per_epoch += batch_loss.item()
        accuracy_epoch += batch_accu.item()
        
    # get return loss and accuracy by dividing by iterator length
    loss = loss_per_epoch/len(iterator)
    accu = accuracy_epoch/len(iterator)
    
    # calculate the AUC score
    guesses_npa = np.asarray(epoch_guesses)
    targets_npa = np.asarray(epoch_targets)
    auc_score = metrics.roc_auc_score(targets_npa, guesses_npa)
    
    return [loss, accu, auc_score]

###############################################################################
    
def evaluator(model, iterator, loss_calculator):
    
    # tell model it is evaluation mode 
    model.eval()
    
    # for each new epoch, these will be reset to 0
    loss_per_epoch = 0
    accuracy_epoch = 0
    
    # initialize two empty lists (to calculate AUC score later)
    epoch_guesses = []
    epoch_targets = []
    
    # deactivate autograd engine (no backprop)
    with torch.no_grad():
    
        for batch in iterator:
        
            # retrieve batch's text examples (padded) and their lengths, respectively
            text_eg, len_txt = batch.text
            
            # get model's predictions
            guess = model(text_eg, len_txt)
            
            # make tensor containing predictions one-dimensional 
            guess = guess.squeeze()
            
            # calculate the batch loss using BCELoss
            batch_loss = loss_calculator(guess, batch.target)
            
            # round predictions to either 1 or 0 
            binary_guess = torch.round(guess)
            
            # correct predictions match the actual targets
            correctGuess = (binary_guess == batch.target).float() 
            
            # append guesses and targets to their respective lists (initialized before loop)
            epoch_guesses.extend(binary_guess.detach().numpy())
            epoch_targets.extend(batch.target.detach().numpy())
            
            # calculate the batch accuracy
            batch_accu = correctGuess.sum()/len(correctGuess)
            
            # update epoch loss & accuracy 
            loss_per_epoch += batch_loss.item()
            accuracy_epoch += batch_accu.item()
        
    # get return loss and accuracy by dividing by iterator length
    loss = loss_per_epoch/len(iterator)
    accu = accuracy_epoch/len(iterator)
    
    # calculate the AUC score
    guesses_npa = np.asarray(epoch_guesses)
    targets_npa = np.asarray(epoch_targets)
    auc_score = metrics.roc_auc_score(targets_npa, guesses_npa)
    
    return [loss, accu, auc_score]

###############################################################################

# will later be converted to numpy arrays
train_graph = []
valid_graph = []

# cycle through the full training data 5 times
for epoch in range(5):
     
    # model training
    train_results = trainer(model, train_iterator, optimizer, get_loss)
    train_graph.append(train_results)
    
    # model validation
    valid_results = evaluator(model, valid_iterator, get_loss)
    valid_graph.append(valid_results)

# convert to numpy arrays
train_npa = np.asarray(train_graph)
valid_npa = np.asarray(valid_graph)

###############################################################################

# plotting code 
import matplotlib.pyplot as plt

plt.figure()
plt.title("Sigmoid: Adam\no <- Training | Validation -> x ")
plt.xlabel("# Epoch")
plt.xticks(range(1,6))

plt.plot(range(1,6), train_npa[:,0], marker='o', color='r')
plt.plot(range(1,6), train_npa[:,1], marker='o', color='b')
plt.plot(range(1,6), train_npa[:,2], marker='o', color='g')

plt.plot(range(1,6), valid_npa[:,0], marker='x', color='r')
plt.plot(range(1,6), valid_npa[:,1], marker='x', color='b')
plt.plot(range(1,6), valid_npa[:,2], marker='x', color='g')

plt.legend(["loss","accuracy","auc score"], loc='best', fancybox=True)

plt.show()

###############################################################################