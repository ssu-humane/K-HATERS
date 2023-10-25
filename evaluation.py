import transformers
import torch
from transformers import *
import glob 
import random
import pandas as pd
from Models.utils import masked_cross_entropy,fix_the_random,format_time,save_normal_model,save_bert_model
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score, precision_recall_curve, auc, multilabel_confusion_matrix
from tqdm import tqdm
from TensorDataset.datsetSplitter import createDatasetSplit
from TensorDataset.dataLoader import combine_features
from Preprocess.dataCollect import collect_data
import matplotlib.pyplot as plt
import time
import os
from sklearn.utils import class_weight
import json
from Models.bertModels import *
from Models.otherModels import *
import sys
from waiting import wait
from sklearn.preprocessing import LabelEncoder
import numpy as np
import argparse
import ast
from Models.utils import *
from fairlearn.metrics import false_positive_rate, true_positive_rate, MetricFrame
import torch.backends.cudnn as cudnn
import pickle
from collections import Counter
from torch import nn
from datasets import load_dataset

def seed_fix(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    
def sigmoid(x):
    return 1/(1+np.exp(-x))

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i]==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1

    return(TP, FP, TN, FN)

def tpr_fpr(true_labels, pred_labels,b_target_label,class_):
    my_metrics = {
        'tpr' : true_positive_rate,
        'fpr' : false_positive_rate
    }
    
    true_labels = (np.array(true_labels)==class_)*1
    pred_labels = (np.array(pred_labels)==class_)*1
    
    label = []
    for i in b_target_label:
        tmp = []
        for j in i:
            if j == -1:
                break
            else:
                tmp.append(j)
        label.append(tmp)

    true_label = []
    pred_label = []
    target_true_label = []
    
    for tl, pl, t in zip(true_labels, pred_labels, label):
        if(len(t)!=1):
            for j in t:
                true_label.append(tl)
                pred_label.append(pl)
                target_true_label.append(j)
        else:
            true_label.append(tl)
            pred_label.append(pl)
            target_true_label.append(t[0])

    mf = MetricFrame(metrics=my_metrics,
                     y_true=true_label,
                     y_pred=pred_label,
                     sensitive_features=target_true_label)
    
    return mf.by_group.index.values, mf.by_group['tpr'].values, mf.by_group['fpr'].values
    
def select_model(params,embeddings):
    model = SC_weighted_BERT.from_pretrained(
        params['path_files'], 
        num_labels = params['num_classes'], 
        output_attentions = True, 
        output_hidden_states = False, 
        hidden_dropout_prob=params['dropout_bert'],
        params=params
    )
    model.to(torch.device('cuda'))
    save_path = "./Saved/"+params['path_files_name']+"_"+params['model_to_use'].replace('_','')+"_"+str(params['train_target'])+"_"+str(params['train_target_att'])+"_"+str(params['train_att'])+"_"+str(params['num_classes'])+".pt"
    model.load_state_dict(torch.load(save_path, map_location='cuda'))
    print("load model...: ", save_path)
    return model

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 


def att(att, mask):
    attention = []
    for A, M in zip(att,mask):
        for a, m in zip(A,M):
            tmp = []
            for i,j in zip(a,m):
                if j:
                    tmp.append(i)
                else:
                    break
            attention.append(np.array(tmp))
    return attention
        
def att_to_bool(true_att):
    true_att_bool = []
    for value in true_att:
        max_v = max(value)
        cnt = Counter(value)[max_v]
        if cnt == len(value):
            true_att_bool.append([0]*len(value))
        else:
            true_att_bool.append([1 if i==max_v else 0 for i in value])
    return true_att_bool

def pred_idx(pred_att):
    pred_att_idx = []

    for value in pred_att:
        top_n = 3
        value = value[1:-1]
        if len(value)<4:
            top_n=1
        idx = value.argsort()[-top_n:]
        pred_att_idx.append(idx[::-1])
    return pred_att_idx

def true_idx(true_att):
    true_att_idx = []

    for value in true_att:
        value = value[1:-1]
        max_v = max(value)
        cnt = Counter(value)[max_v]
        if cnt == len(value):
            true_att_idx.append([])
        else:
            true_att_idx.append([i for i,c in enumerate(value) if c==max_v])
    return true_att_idx

def true_span(true_idx):
    true_ = []
    for j in true_idx:
        if len(j)==0:
            true_.append([])
            continue
        tmp_list = []
        tmp = j[0]
        cnt = 1
        for i in j[1:]:
            if i==(tmp+cnt):
                cnt+=1
            else:
                tmp_list.append((tmp, tmp+cnt))
                tmp=i
                cnt=1
        tmp_list.append((tmp, tmp+cnt))
        true_.append(tmp_list)
    return true_
        
        
def Eval_phase(params,encoder,target_encoder,true_labels,pred_labels,logits_all_final,target_pred_labels,target_pred_label,target_true_labels,target_logits_all,target_true_label,target_labels,target_true_label_,pred_att_all,true_att_all,att_mask_all):
    testf1=f1_score(true_labels, pred_labels, average='macro')
    testf1_micro=f1_score(true_labels, pred_labels, average='micro')
    testf1_class=f1_score(true_labels, pred_labels, average=None)
    testacc=accuracy_score(true_labels,pred_labels)
    testrocauc=roc_auc_score(true_labels, logits_all_final, multi_class='ovo', average='macro') 
    testprecision=precision_score(true_labels, pred_labels, average='macro')
    testprecision_ = precision_score(true_labels, pred_labels, average=None)
    testrecall=recall_score(true_labels, pred_labels, average='macro')
    testrecall_=recall_score(true_labels, pred_labels, average=None)

    tp_=0
    fp_=0
    tn_=0
    fn_=0

    for i in range(params['num_classes']):
        tp,fp,tn,fn = perf_measure((np.array(true_labels)==i)*1, (np.array(pred_labels)==i)*1)
        tp_+=tp
        fp_+=fp
        tn_+=tn
        fn_+=fn
    whole_tpr = tp_/(tp_+fn_)
    whole_fpr = fp_/(fp_+tn_)
    
    print("\n--------------hate--------------")
    print(" Accuracy: {0:.5f}".format(testacc))
    print(" Fscore (macro): {0:.5f}".format(testf1))
    print(" class Fscore:", testf1_class)
    print(" Precision: {0:.5f}".format(testprecision))
    print(" Precision_: ", testprecision_)
    print(" Recall: {0:.5f}".format(testrecall))
    print(" Recall_: ",testrecall_)
    print(" Roc Auc: {0:.5f}".format(testrocauc))
        
        
    target_logits_all_final=[]
    for logits in target_logits_all:
        target_logits_all_final.append(softmax(logits))

    target_testacc=accuracy_score(target_true_labels,target_pred_labels)
    target_testf1=f1_score(target_true_label, target_pred_label, average='macro')
    target_testf1_class = f1_score(target_true_label, target_pred_label, average=None)
    target_testprecision=precision_score(target_true_label, target_pred_label, average='macro')
    target_testprecision_=precision_score(target_true_label, target_pred_label, average=None)
    target_testrecall=recall_score(target_true_label, target_pred_label, average='macro')
    target_testrecall_=recall_score(target_true_label, target_pred_label, average=None)

    print("\n--------------Target--------------")
    print(" target Accuracy: {0:.5f}".format(target_testacc))
    print(" target Fscore (macro): {0:.5f}".format(target_testf1))
    print(" target class Fscore: ", target_testf1_class)
    print(" target Precision: {0:.5f}".format(target_testprecision))
    print(" target Precision_: ", target_testprecision_)
    print(" target Recall: {0:.5f}".format(target_testrecall))
    print(" target Recall_: ", target_testrecall_)


    print(multilabel_confusion_matrix(target_true_label,target_pred_label))
    tpr_fpr_list = []
    
    ### fairlean
    print("\n--------------fairlearn--------------")
    for i in range(params['num_classes']):
        tmp = []
        idx, tpr, fpr = tpr_fpr(true_labels, pred_labels, target_labels, i) 
        if i==0:
            print("target classes: ", target_encoder.inverse_transform(idx))
        print("-------",encoder.inverse_transform([i])[0],"'s tpr & fpr-------")
        print('tpr: ', tpr)
        print('fpr: ', fpr)
        tmp.append(encoder.inverse_transform([i])[0])
        tmp.extend(tpr)
        tpr = [j for j in tpr if j!=0]
        try:
            tpr_diff = max(tpr)-min(tpr)
        except:
            tpr_diff = 0
        tmp.extend(fpr)
        fpr_diff = max(fpr)-min(fpr)
        tmp.extend([tpr_diff, fpr_diff])
        tpr_fpr_list.append(tmp)
          
    all_auc = []
    all_token_f1 = []
    all_iou_f1 = []
    
    ex_rationale = []
    for n,i in enumerate(true_att_all):
        for t in i:
            if 1 in t:
                ex_rationale.append(n)
    with open("faithfulness/rationale_ex_idx.pickle",'wb') as f:
        pickle.dump(ex_rationale,f)

    
    n = 12
    for head_num in range(n):
        true = att(true_att_all, att_mask_all)
        true = att_to_bool(true)
        pred = att([i[:,head_num,0,:] for i in pred_att_all], att_mask_all)
        aucs = []
        for k, (t, p) in enumerate(zip(true, pred)):
            if k in ex_rationale:
                if sum(t)==0: ######!#######
                    continue
                precision, recall, _ = precision_recall_curve(t, p)
                aucs.append(auc(recall, precision))
        att_auc = np.average(aucs)
        print('--------------------------auprc is {0:.5f}--------------------------'.format(att_auc))

        f1_list = []
        p_idx = pred_idx(pred)
        t_idx = true_idx(true)
        
        with open("faithfulness/predicted_idx_"+str(head_num)+".pickle",'wb') as f:
            pickle.dump(p_idx,f)

        for k, (i,j) in enumerate(zip(p_idx, t_idx)):
            if k in ex_rationale:
                if len(i)>0:
                    instance_prec = len(set(i)&set(j))/len(i)
                else:
                    instance_prec = 0
                if len(j)>0:
                    instance_rec = len(set(i)&set(j))/len(j)
                else:
                    instance_rec = 0
                f1_list.append(0 if instance_prec==0 or instance_rec==0 else (2*instance_rec*instance_prec)/(instance_rec+instance_prec))
        token_f1 = sum(f1_list)/len(f1_list)
        print('--------------------------token f1 is {0:.5f}--------------------------'.format(token_f1))

        t_span = true_span(t_idx)
        ious = []
        for k, (pred, true) in enumerate(zip(p_idx, t_span)):
            iou_tmp = []
            if k in ex_rationale:
                for p in pred:
                    best_iou = 0.0
                    for t in true:
                        if type(p)!=list:
                            p = [p]
                        num = len(set(p) & set(range(t[0],t[1])))
                        denom = len(set(p) | set(range(t[0],t[1])))
                        iou = 0 if denom == 0 else num / denom
                        if iou > best_iou:
                            best_iou = iou
                    iou_tmp.append(best_iou)
                ious.append(iou_tmp)

        threshold_tps = []
        iou_threshold = 0.5
        for vs in ious:
            threshold_tps.append(sum(int(x >= iou_threshold) for x in vs))

        macro_rs = list(j / len(i) if len(i) > 0 else 0 for i,j in zip(t_span,threshold_tps))
        macro_ps = list(j / len(i) if len(i) > 0 else 0 for i,j in zip(p_idx,threshold_tps))
        macro_r = sum(macro_rs) / len(macro_rs) if len(macro_rs) > 0 else 0
        macro_p = sum(macro_ps) / len(macro_ps) if len(macro_ps) > 0 else 0
        iou_f1 = 0 if macro_r==0 or macro_p==0 else 2*macro_r*macro_p/(macro_r+macro_p)
        print('--------------------------iou f1 is {0:.5f}--------------------------'.format(iou_f1))
        all_auc.append(att_auc)
        all_token_f1.append(token_f1)
        all_iou_f1.append(iou_f1)
    
        
    return all_auc, all_token_f1, all_iou_f1


def comp_suff(params,comp_flag):
    with open("faithfulness/before_comp_prob.pickle", "rb") as f:
        before = pickle.load(f)
    with open("faithfulness/comp_prob.pickle", "rb") as f:
        after = pickle.load(f)
    with open("faithfulness/rationale_ex_idx.pickle",'rb') as f:
        ex_rationale = pickle.load(f)
        
    tmp = []
    for k,(i,j) in enumerate(zip(before, after)):
        if k in ex_rationale:
            idx = i.argmax()
            comp = i[idx]-j[idx]
            tmp.append(comp)
    v = np.average(tmp)
    v_name = 'comprehensiveness' if comp_flag else 'sufficiency'
    print('--------------------------{0} is {1:.5f}--------------------------'.format(v_name,v))
    
    if params['train_att']:
        model_name = params['model_to_use']+"_"+str(params['train_target'])+"_"+str(params['train_target_att'])+"_"+str(params['train_att'])+"_"+str(params['num_supervised_heads'])

        file_name = "Saved/tmp_result.csv"
        df = pd.read_csv(file_name)

        if comp_flag:
            df.at[len(df)-1,'explainability (comp)']= v
        else:
            df.at[len(df)-1,'explainability (suff) ↓'] = v

        df.to_csv(file_name, encoding='utf-8-sig', index=False)
    else:
        return v
    
def measurement(params,comp,suff,which_files='test',model=None,test_dataloader=None,device=None):
    model.eval()

    print("Running eval on test...")
    t0 = time.time()
    true_labels=[]
    pred_labels=[]
    logits_all=[]
    target_pred_labels=[]
    target_pred_label=[]
    target_true_labels=[]
    target_logits_all=[]
    target_true_label=[]
    target_labels=[]
    target_true_label_=[]
    pred_att_all = []
    true_att_all = []
    att_mask_all = []
    threshold = 0.5
    if (comp or suff):
        with open("faithfulness/predicted_idx_"+str(params['head_num'])+".pickle", 'rb') as f:
            pred_att_idx = pickle.load(f)
    with torch.no_grad():
        for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), ascii=True):

            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)

            if (comp or suff):
                b_input_ids_tmp = []
                b_input_mask_tmp = []
                for idx, ids, mask in zip(pred_att_idx[step*params['batch_size']:step*params['batch_size']+params['batch_size']],batch[0],batch[2]):
                    tmp = []
                    tmp_ = []
                    idx = [j+1 for j in idx]
                    for n, (i,j) in enumerate(zip(ids,mask)):
                        if comp:
                            if n not in idx:
                                tmp.append(i)
                                tmp_.append(j)
                            else:
                                tmp.append(1)
                                tmp_.append(1)
                        elif suff:
                            if n in idx:
                                tmp.append(i)
                                tmp_.append(j)
                            else:
                                tmp.append(1)
                                tmp_.append(1)
                    b_input_ids_tmp.append(tmp)
                    b_input_mask_tmp.append(tmp_)
                
                b_input_ids = torch.tensor(b_input_ids_tmp).to(device)
                b_input_mask = torch.tensor(b_input_mask_tmp).to(device)
            else:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[2].to(device)
            b_att_val = batch[1].to(device)
            b_labels = batch[3].to(device)
            b_target_att = batch[4].to(device)
            b_target_labels = batch[5].to(device)
            model.zero_grad()
            outputs = model(b_input_ids,
                attention_vals=b_att_val,
                attention_mask=b_input_mask.to(torch.bool),
                labels=None,
                target_att=b_target_att,
                target_labels=b_target_labels,
                device=device)

            n = 2 
            num_supervised_heads = params['num_supervised_heads']*2
            attention_vector=outputs[n][params['supervised_layer_pos']].detach().cpu().numpy()
            pred_att_all.append(attention_vector)
            true_att_all.append(b_att_val.detach().cpu().numpy())
            att_mask_all.append(b_input_mask.detach().cpu().numpy())
            
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            pred_labels+=list(np.argmax(logits, axis=1).flatten())
            true_labels+=list(label_ids.flatten())
            logits_all+=list(logits)
            target_labels += b_target_labels.cpu().tolist()
            
            target_logits = outputs[1]
            target_logits = target_logits.detach().cpu().numpy()
            label = []
            for i in b_target_labels:
                tmp = [0]*params['target_num_classes']
                for j in i:
                    if j == -1:
                        break
                    else:
                        tmp[j]=1
                label.append(tmp)

            for i in target_logits:
                target_pred_labels+=((sigmoid(np.array(i))>threshold )*1).tolist()
                target_pred_label.append(((sigmoid(np.array(i))>threshold )*1).tolist())

            target_true_label += label 
            target_true_labels += sum(label, []) 
            target_logits_all += list(target_logits)
                
                
    logits_all_final=[]
    for logits in logits_all:
        logits_all_final.append(softmax(logits))
    if (comp or suff):
        with open("faithfulness/comp_prob.pickle",'wb') as f:
            pickle.dump(logits_all_final,f)
    else:
        with open("faithfulness/before_comp_prob.pickle",'wb') as f:
            pickle.dump(logits_all_final,f)
    if (comp or suff)==False:
        with open("pred_labels.pickle",'wb') as f:
            pickle.dump(pred_labels,f)
        with open("true_labels.pickle",'wb') as f:
            pickle.dump(true_labels,f)
            
    return true_labels,pred_labels,logits_all_final,target_pred_labels,target_pred_label,target_true_labels,target_logits_all,target_true_label,target_labels,target_true_label_,pred_att_all,true_att_all,att_mask_all


def train_model(params,device):
    embeddings=None

#     data = load_dataset("humane-lab/K-HATERS")
#     test = data['test']
    
    with open("Data/Total_data_4/test_data.pickle", 'rb') as f:
        test = pickle.load(f)
        
    target_encoder = LabelEncoder()
    target_encoder.fit(params['target_class_name'])
    
    encoder = LabelEncoder()
    encoder.fit(np.load(params['class_names'],allow_pickle=True))
    
    test_dataloader=combine_features(test, params, is_train=False)
    
    model=select_model(params,embeddings)
        
        
    if(params["device"]=='cuda'):
        model.cuda()
        
    optimizer = AdamW(model.parameters(),
                  lr = params['learning_rate'],
                  eps = params['epsilon']
                )


    total_steps = len(test_dataloader) * params['epochs']

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(total_steps/10), num_training_steps = total_steps)
   
    true_labels,pred_labels,logits_all_final,target_pred_labels,target_pred_label,target_true_labels,target_logits_all,target_true_label,target_labels,target_true_label_,pred_att_all,true_att_all,att_mask_all=measurement(params,False,False,'test',model,test_dataloader,device)
    all_auc, all_token_f1, all_iou_f1 = Eval_phase(params,encoder,target_encoder,true_labels,pred_labels,logits_all_final,target_pred_labels,target_pred_label,target_true_labels,target_logits_all,target_true_label,target_labels,target_true_label_,pred_att_all,true_att_all,att_mask_all)
    
    
    comp = []
    suff = []
    for i in range(12):
        params['head_num']=i
        measurement(params,True,False,'test',model,test_dataloader,device)
        comp.append(comp_suff(params,True))

        measurement(params,False,True,'test',model,test_dataloader,device)
        suff.append(comp_suff(params,False))

    tmp = [i+j for i,j in zip(all_iou_f1, comp)]
    argmax = np.argmax(tmp)
    print("가장 좋은 explainability를 갖는 head: ",argmax)
    print("explainability (comp): ",comp[argmax])
    print("explainability (suff) ↓: ",suff[argmax])
    print("explainability (iou_f1): ",all_iou_f1[argmax])
    print("explainability (token_f1): ",all_token_f1[argmax])
    print("explainability (auprc): ",all_auc[argmax])
        
    

if __name__=='__main__': 
    my_parser = argparse.ArgumentParser(description='Which model to use')
    my_parser.add_argument('seed',
                           metavar='--seed',
                           type=int,
                           help='The model to use')
    
    args = my_parser.parse_args()
    seed = args.seed
    with open('best_model_json/bestModel_bert_base_uncased_Attn_train_FALSE.json', mode='r') as f:
        params = json.load(f)
    params['best_params']=True 
    
    for key in params:
        if params[key] == 'True':
             params[key]=True
        elif params[key] == 'False':
             params[key]=False
        if (key in ['batch_size','num_classes','hidden_size','supervised_layer_pos','max_length','epochs','variance','num_classes','num_supervised_heads','num_supervised_heads']):
            if(params[key]!='N/A'):
                params[key]=int(params[key])
                
    torch.autograd.set_detect_anomaly(True)
    
    if torch.cuda.is_available() and params['device']=='cuda':    
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    params['data_file']='Data/data_processed_4.pickle'
    params['class_names']='Data/classes_four.npy' 
    params['head_num']=0
    params['inference']=False
    params['multi'] = 'multi-label'
    
    params['target_class_name'] = np.array(['gender','age','region','disabled','religion','political','job','individual','others'])
    params['target_num_classes'] = len(params['target_class_name'])
    
    params['num_supervised_heads']=params['num_supervised_heads']*2
    params['train']=False
    seed_fix(seed)
    train_model(params,device)
