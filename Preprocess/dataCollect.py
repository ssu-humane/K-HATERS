import pandas as pd
from glob import glob
import json
from tqdm import tqdm_notebook,tqdm
from difflib import SequenceMatcher
from collections import Counter
from transformers import BertTokenizer, XLMRobertaTokenizer
from transformers import RobertaTokenizer
from transformers import AutoModel, AutoTokenizer
from os import path
import pickle
import numpy as np
import re
import torch
import torch.backends.cudnn as cudnn
import random
import ast

def set_name(params):
    file_name='Data/Total_data'
    file_name+=params['data_file'].replace('Data/data_processed','')
    return file_name



def get_annotated_data(params):
    data = pd.read_pickle(params['data_file'])
    dict_data=[]
    for i in data.iloc():
        temp={}
        temp['text'] = i['token']
        temp['token_ids'] = i['token_ids']
        temp['rationales'] = i['rationales_']
        temp['target_rationales'] = i['t_rationales_']
        temp['final_label'] = i['final_label']
        temp['target_label']=i['target']
        if(temp['target_label']==[]):
            temp['target_rationales'] = [0]*len(temp['target_rationales'])
        dict_data.append(temp) 
        
    temp_read = pd.DataFrame(dict_data) 
    return temp_read   




def get_training_data(data,params,tokenizer):
    text_list=[]
    attention_list=[]
    targetAttention_list=[]
    label_list=[]
    target_label_list=[]
    count_confused=0
    print('total_data',len(data))
    for index,row in tqdm(data.iterrows(),total=len(data), ascii=True):
        text=row['text']

        annotation=row['final_label']
        target_annotation=row['target_label']
        attention_masks = row['rationales']
        attention_masks_ = row['target_rationales']
        tokens_all = row['token_ids']
        
        if (sum(attention_masks) == 0): 
            attention_vector = [0 for x in tokens_all]
        else:
            attention_vector = attention_masks
            
        if (sum(attention_masks_) == 0):
            attention_vector_ = [0 for x in tokens_all]
        else:
            attention_vector_ = attention_masks_

        attention_list.append(attention_vector)
        targetAttention_list.append(attention_vector_)
        text_list.append(tokens_all)
        label_list.append(annotation)
        target_label_list.append(target_annotation)
    training_data = pd.DataFrame(list(zip(text_list,attention_list,targetAttention_list,label_list, target_label_list)), 
                   columns =['Text', 'Attention' ,'target_Attention', 'Label', 'target_Label']) 
    
    
    filename=set_name(params)
    training_data.to_pickle(filename)
    return training_data


def collect_data(params):
    tokenizer = AutoTokenizer.from_pretrained(params['path_files'], do_lower_case=False)
    data_all_labelled=get_annotated_data(params)
    train_data=get_training_data(data_all_labelled,params,tokenizer)
    return train_data