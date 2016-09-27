# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 15:09:15 2016

@author: rahma
"""

from __future__ import division
import numpy as np
import glob #to read all the .npy of a directory
import os
import pandas as pd


EMBEDDING_PATH = '/home/rchaabouni/data/BUCKEYE_embeddings/embeddings' 
CORPUS_PATH = '/fhgfs/bootphon/data/derived_data/BUCKEYE_revised_bootphon'

# read the embeddings
os.chdir(EMBEDDING_PATH)

numpy_vars = {}

string_list = []    
for np_name in glob.glob('*.npy'):
    tmp = np_name.split('.')[0]
    string_list.append(tmp)
    numpy_vars[tmp] = np.load(np_name)
    
# read the corpus
end_time = {}
phones = {}
for np_name in string_list:
    file_path = os.path.join(CORPUS_PATH,
                 np_name[0:3], np_name, np_name+'.phones')
    end_time[np_name] = []
    phones[np_name] = []             
    with open(file_path) as f:
        for _ in xrange(9):
            next(f)
        for line in f:
            splitting = line.split()
            end_time[np_name].append(splitting[0])
            phones[np_name].append(splitting[2])                
#%%
########################################################
#           Creating a vector for each phone           #
#######################################################
freq = 100 #embedding sampling frequency
phone_embedding ={}

for np_name in string_list:
    embedding = numpy_vars[np_name]
    time_1 = end_time[np_name]
    time_1 = [float(x) for x in time_1]
    time_1 = [0] + list(time_1)
    time = [(a + b) / 2 for a, b in zip(time_1[0::1], time_1[1::1])]
    sample_index = [int(freq* x) for x in time]

    for cc, index in enumerate(sample_index):
        vector =[]
        if (index>=20) and (index<= len(embedding)-20):
            for counter in range(index-20,index+20):
                vector = vector + list(embedding[counter])
            
            if not(phones[np_name][cc] in phone_embedding.keys()):
                phone_embedding[phones[np_name][cc]] = []
            phone_embedding[phones[np_name][cc]].append(vector)

#%%        
############################################
#### average the vectors representation ####
############################################     
for key in phone_embedding.keys():
    phone_embedding[key] = np.mean(phone_embedding[key], axis = 0) 
    
    
#%%
# create csv file
table = pd.DataFrame(data = phone_embedding.values(), index = phone_embedding.keys())    
#table.to_csv('/home/rchaabouni/data/BUCKEYE_embeddings/embedding_evaluation.csv')