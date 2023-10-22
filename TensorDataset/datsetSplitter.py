import sys, os
from tqdm import tqdm
import numpy as np
import sys, os
sys.path.append('../')
from torch.utils.data import Dataset
import pandas as pd
from Preprocess.dataCollect import collect_data,set_name
from sklearn.model_selection import train_test_split
from os import path
from gensim.models import KeyedVectors
import pickle
import json
import gensim
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from collections import Counter
import random
    
def encodeData(dataframe,vocab,tokenizer,params):
    tuple_new_data=[]
    for index,row in tqdm(dataframe.iterrows(),total=len(dataframe), ascii=True):
        tuple_new_data.append((row['Text'],row['Attention'],row['Label'],row['target_Attention'],row['target_Label']))
    return tuple_new_data



def createDatasetSplit(params):
    filename=set_name(params)
    if path.exists(filename):
        pass
    else:
        dataset = collect_data(params)
    if path.exists(filename[:-7]+'/train_data.pickle'):
        with open(filename[:-7]+'/train_data.pickle', 'rb') as f: 
            X_train = pickle.load(f)
        with open(filename[:-7]+'/val_data.pickle', 'rb') as f:
            X_val = pickle.load(f)
        with open(filename[:-7]+'/test_data.pickle', 'rb') as f:
            X_test = pickle.load(f)
    else:
        dataset = pd.read_pickle(filename)
        
        X_train_dev, X_test= train_test_split(dataset, test_size=10000, random_state=0, stratify=dataset['Label'])
        X_train, X_val= train_test_split(X_train_dev, test_size=10000, random_state=0, stratify=X_train_dev['Label'])
        
        vocab_own=None    
        tokenizer = AutoTokenizer.from_pretrained(params['path_files'], do_lower_case=False)
        
        X_train = encodeData(X_train,vocab_own,tokenizer,params) # balanced sampling 전 train 데이터
        X_val=encodeData(X_val,vocab_own,tokenizer,params)
        X_test=encodeData(X_test,vocab_own,tokenizer,params)
        
        try:
            os.mkdir(filename[:-7])
        except:
            pass
            
        with open(filename[:-7]+'/train_data.pickle', 'wb') as f:
            pickle.dump(X_train, f)
        with open(filename[:-7]+'/val_data.pickle', 'wb') as f:
            pickle.dump(X_val, f)
        with open(filename[:-7]+'/test_data.pickle', 'wb') as f:
            pickle.dump(X_test, f)
            
    return X_train,X_val,X_test
              
