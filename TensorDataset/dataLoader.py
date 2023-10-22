import torch
import transformers
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

def custom_att_masks(input_ids, path_files_name):
    attention_masks = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return attention_masks

def combine_features(tuple_data,params,is_train=False):
    input_ids = [ele[0] for ele in tuple_data]
    att_vals = [ele[1] for ele in tuple_data]
    labels = [ele [2] for ele in tuple_data]
    target_att_vals = [ele [3] for ele in tuple_data]
    target_labels = [ele [4] for ele in tuple_data]


    encoder = LabelEncoder()
    encoder.fit(np.load(params['class_names'],allow_pickle=True))
    labels=encoder.transform(labels)

    target_encoder = LabelEncoder()
    target_encoder.fit(params['target_class_name'])
    if is_train==True:
        print(target_encoder.classes_)
    tmp=[]
    tmp2=[]
    for i in target_labels:
        tmp.append(target_encoder.transform(i).tolist())

    for i in tqdm(tmp):
        tmp2.append(i + [-1]*(params['target_num_classes'] - len(i)))
    target_labels = tmp2


    att_vals = pad_sequences(att_vals,maxlen=int(params['max_length']), dtype="float", value=0.0, truncating="post", padding="post")
    target_att_vals = pad_sequences(target_att_vals,maxlen=int(params['max_length']), dtype="float", value=0.0, truncating="post", padding="post")
    input_ids = pad_sequences(input_ids, maxlen=int(params['max_length']), dtype="long", value=0, truncating="post",
                              padding="post")


    att_masks=custom_att_masks(input_ids, params['path_files_name'])
    dataloader = return_dataloader(input_ids,labels,att_vals,att_masks,target_att_vals,target_labels,params,is_train)
    return dataloader

def return_dataloader(input_ids,labels,att_vals,att_masks,target_att_vals,target_labels,params,is_train=False):
    inputs = torch.tensor(input_ids)
    labels = torch.tensor(labels,dtype=torch.long)
    target_labels = torch.tensor(target_labels,dtype=torch.long)
    masks = torch.tensor(np.array(att_masks),dtype=torch.uint8)
    attention = torch.tensor(np.array(att_vals),dtype=torch.float)
    target_attention = torch.tensor(np.array(target_att_vals),dtype=torch.float)
    data = TensorDataset(inputs,attention,masks,labels,target_attention, target_labels)
    
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=params['batch_size'], shuffle=False)
    return dataloader

