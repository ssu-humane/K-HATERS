from tqdm.auto import tqdm
import transformers 
from transformers import *
import torch
from torch import nn
import pandas as pd
import numpy as np
from Models.utils import * 
from Models.bertModels import *
from TensorDataset.dataLoader import combine_features
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
import os
import json
import time
import random
from sklearn.preprocessing import LabelEncoder
import torch.backends.cudnn as cudnn
from datasets import load_dataset


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
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

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def select_model(params,embeddings):
    model = SC_weighted_BERT.from_pretrained(
        params['path_files'],
        num_labels = params['num_classes'],
        output_attentions = True, 
        output_hidden_states = False, 
        hidden_dropout_prob=params['dropout_bert'],
        params=params
    )
    
    return model


def Eval_phase(params,which_files='test',model=None,test_dataloader=None,device=None):
    print("Running eval on ",which_files,"...")
    t0 = time.time()
    
    true_labels=[]
    pred_labels=[]
    logits_all=[]
    
    target_pred_labels=[]
    target_true_labels=[]
    target_logits_all=[]
    target_pred_label=[]
    target_true_label=[]
    total_loss=0
    threshold=0.5
    loss_funct = CrossEntropyLoss()

    with torch.no_grad():
        for step, batch in tqdm(enumerate(test_dataloader), ascii=True):

            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)


            b_input_ids = batch[0].to(device)
            b_att_val = batch[1].to(device)
            b_input_mask = batch[2].to(device)
            b_labels = batch[3].to(device)
            b_target_att = batch[4].to(device)
            b_target_labels = batch[5].to(device)



            model.zero_grad()
            outputs = model(b_input_ids,
                attention_vals=b_att_val,
                attention_mask=b_input_mask.to(torch.bool),
                labels=None,target_labels=None,device=device)
            
            logits = outputs[0]
            loss = focal_multi_class(logits.view(-1, params['num_classes']), b_labels.view(-1))
            target_logits = outputs[1]
            label = []
            for i in b_target_labels:
                tmp = [0]*params['target_num_classes']
                for j in i:
                    if j == -1:
                        break
                    else:
                        tmp[j]=1
                label.append(tmp)
            label = torch.tensor(label).to(device)

            loss +=  focal_binary_cross_entropy(target_logits, label, params['target_num_classes'])
            total_loss+=loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            
            pred_labels+=list(np.argmax(logits, axis=1).flatten())
            true_labels+=list(label_ids.flatten())
            logits_all+=list(logits)
            
            target_logits = target_logits.detach().cpu().numpy()
            for i in target_logits:
                target_pred_labels+=((sigmoid(np.array(i))>0.5)*1).tolist() 
                target_pred_label.append(list(((sigmoid(np.array(i))>threshold )*1)))
            target_true_labels+=sum(label.cpu().tolist(), [])
            target_true_label += label.cpu().tolist()

            target_logits_all+=list(target_logits)
    
    
    
    logits_all_final=[]
    for logits in logits_all:
        logits_all_final.append(softmax(logits))
        

    # hate
    print("avg_val_loss: ", total_loss/len(test_dataloader)) 
    testf1=f1_score(true_labels, pred_labels, average='macro')
    testf1_class = f1_score(true_labels, pred_labels, average=None)
    testacc=accuracy_score(true_labels,pred_labels)
    if(params['num_classes']==2):
        l=[]
        for i in logits_all_final:
            l.append(i[1])
        testrocauc=roc_auc_score(true_labels, l) 
    else:
        testrocauc=roc_auc_score(true_labels, logits_all_final, multi_class='ovo', average='macro') 
    testprecision=precision_score(true_labels, pred_labels, average='macro')
    testprecision_=precision_score(true_labels, pred_labels, average=None)
    testrecall=recall_score(true_labels, pred_labels, average='macro')
    testrecall_=recall_score(true_labels, pred_labels, average=None)
    
    print(" Accuracy: {0:.5f}".format(testacc))
    print(" Fscore: {0:.5f}".format(testf1))
    print(" class Fscore:", testf1_class)
    print(" Precision: {0:.5f}".format(testprecision))
    print(" Precision_: ", testprecision_)
    print(" Recall: {0:.5f}".format(testrecall))
    print(" Recall_: ",testrecall_)
    print(" Roc Auc: {0:.5f}".format(testrocauc))
        
    # target
    target_logits_all_final=[]
    for logits in target_logits_all:
        target_logits_all_final.append(softmax(logits))
    testacc=accuracy_score(target_true_labels,target_pred_labels)
    testf1=f1_score(target_true_label, target_pred_label, average='macro')
    testf1_class = f1_score(target_true_label, target_pred_label, average=None)
    print("------------Target---------------\n")
    if(params['is_model'] == True):
        print(" target Accuracy: {0:.5f}".format(testacc))
        print(" target Fscore: {0:.5f}".format(testf1))
        print(" target class Fscore:", testf1_class)

    return testf1,testf1_class,testacc,testprecision,testprecision_,testrecall,testrecall_,testrocauc,logits_all_final
    
def interval_eval(params,which_files='test',model=None,test_dataloader=None,device=None):
    model.eval()

    total_loss=0

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):

            b_input_ids = batch[0].to(device)
            b_att_val = batch[1].to(device)
            b_input_mask = batch[2].to(device)
            b_labels = batch[3].to(device)
            b_target_att = batch[4].to(device)
            b_target_labels = batch[5].to(device)


            model.zero_grad()
            outputs = model(b_input_ids,
                attention_vals=b_att_val,
                attention_mask=b_input_mask.to(torch.bool),
                labels=b_labels,target_att=b_target_att,
                target_labels=b_target_labels,device=device)
            
            loss = outputs[0]
            
            total_loss += loss.item()
            
    val_loss = total_loss/len(test_dataloader)
    
    return val_loss
    
def train_model(params,device):
    embeddings=None
    data = load_dataset("humane-lab/K-HATERS")

    train = data['train']
    val = data['validation']
    test = data['test']
    
    train_dataloader=combine_features(train,params,is_train=True)
    validation_dataloader=combine_features(val,params,is_train=False)
    test_dataloader=combine_features(test,params,is_train=False)
   
    model=select_model(params,embeddings)
    
    if(params["device"]=='cuda'):
        model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(),
                  lr = params['learning_rate'],
                  eps = params['epsilon'] 
                )


    total_steps = len(train_dataloader) * params['epochs']

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(total_steps/10), num_training_steps = total_steps)

    loss_values = []
    
    train_loss = []
    train_losses = []
    val_losses = []
    
    att_train_loss = []
    att_train_losses = []
    losses = AverageMeter()
    
    cnt = 0
    patience = 5
    log_interval = len(train_dataloader)//10

    for epoch_i in range(0, params['epochs']):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, params['epochs']))
        print('Training...')

        t0 = time.time()

        total_loss = 0
        
        model.train()
        tbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), ascii=True)
        for step, batch in tbar:
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)

            b_input_ids = batch[0].to(device)
            b_att_val = batch[1].to(device)
            b_input_mask = batch[2].to(device)
            b_labels = batch[3].to(device)
            b_target_att = batch[4].to(device)
            b_target_labels = batch[5].to(device)
            model.zero_grad()
            
            outputs = model(b_input_ids,
                            attention_vals=b_att_val,
                            attention_mask=b_input_mask.to(torch.bool),
                            labels=b_labels,
                            device=device,
                            target_att=b_target_att,
                            target_labels=b_target_labels)
            loss = outputs[0]
            
            losses.update(loss.item(), params['batch_size'])

            tbar.set_description("loss: {0:.6f}".format(losses.avg), refresh=True)
            
            train_loss.append(loss.item())
            
            total_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            scheduler.step()
            if step % log_interval == 0:
                train_losses.append(np.average(train_loss))
                train_loss = []
                
                val_loss = interval_eval(params,'val',model,validation_dataloader,device)

                if len(val_losses)<1:
                    val_losses.append(val_loss)
                    pass
                elif val_losses[-1] < val_loss:
                    cnt+=1
                    if cnt==patience:
                        break
                else:
                    val_losses.append(val_loss)
                    cnt = 0 
                    save_path = "./Saved/"+params['path_files_name']+"_"+params['model_to_use'].replace('_','')+"_"+str(params['train_target'])+"_"+str(params['train_target_att'])+"_"+str(params['train_att'])+"_"+str(params['num_classes'])+".pt"
                    if os.path.exists("./Saved"):
                        pass
                    else:
                        os.mkdir("Saved")
                    torch.save(model.state_dict(), save_path)
        avg_train_loss = total_loss / len(train_dataloader)
        print('avg_train_loss: ',avg_train_loss)

        loss_values.append(avg_train_loss)
        if cnt==patience:
            break
        val_fscore,val_fscore_class,val_accuracy,val_precision,val_precision_,val_recall,val_recall_,val_roc_auc,_=Eval_phase(params,'val',model,validation_dataloader,device)

    del model
    torch.cuda.empty_cache()
    return 1




if __name__=='__main__': 
    with open('best_model_json/bestModel_bert_base_uncased_Attn_train_FALSE.json', mode='r') as f:
        params = json.load(f)
    for key in params:
        if params[key] == 'True':
             params[key]=True
        elif params[key] == 'False':
             params[key]=False
        if (key in ['batch_size','num_classes','hidden_size','supervised_layer_pos','random_seed','max_length','epochs','variance','num_supervised_heads']):
            if(params[key]!='N/A'):
                params[key]=int(params[key])
            
    params['best_params']=True
    torch.autograd.set_detect_anomaly(True)
    if torch.cuda.is_available() and params['device']=='cuda':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        

    params['inference']=False
    params['class_names']='Data/classes_four.npy'
    params['num_supervised_heads']=params['num_supervised_heads']*2
    params['train'] = True
    encoder = LabelEncoder()
    encoder.fit(np.load(params['class_names'],allow_pickle=True))
    
    params['target_class_name'] = np.array(['gender','age','region','disabled','religion','political','job','individual','others'])
    params['target_num_classes'] = len(params['target_class_name'])

    params['random_seed'] = 0
    seed_fix(params['random_seed'])
    train_model(params,device)