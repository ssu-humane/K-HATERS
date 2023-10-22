from transformers.models.bert.modeling_bert import *
from torch import nn
from transformers import AutoModel, AutoConfig, XLMRobertaModel,AutoModelForMaskedLM,RobertaModel
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import json
from .utils import *

with open('best_model_json/bestModel_bert_base_uncased_Attn_train_FALSE.json', mode='r') as f:
    params = json.load(f)

class SC_weighted_BERT(BertPreTrainedModel):
    def __init__(self, config,params):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.target_num_labels = params['target_num_classes']
        self.weights=params['weights']
        self.train_att= params['train_att']
        self.lam = params['att_lambda'] 
        self.alpha = params['hate_alpha']
        self.beta = params['target_beta']
        self.num_sv_heads=params['num_supervised_heads']
        self.sv_layer = params['supervised_layer_pos']
        self.train_target = params['train_target']
        self.train_target_att = params['train_target_att']
        self.model_path = params['path_files']
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier_target = nn.Linear(config.hidden_size, params['target_num_classes'])
        self.softmax=nn.Softmax(config.num_labels)
        self.init_weights() # 초기화 되지 않은 layer에 대해 초기화
 
    def forward(self,
        input_ids=None,
        attention_mask=None,
        attention_vals=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        device=None,
        target_att=None,
        target_labels=None):
        
        device = torch.device("cuda")
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1] 

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output) # hate classification
        if(self.train_target):
            logits_target = self.classifier_target(pooled_output) # target classification
            outputs = (logits, logits_target) + outputs[2:]
        else:
            outputs = (logits,) + outputs[2:]
            
        n = 2 if self.train_target else 1
        if labels is not None:
            label_loss = 0
            loss_funct = CrossEntropyLoss()# weight=torch.tensor(self.weights).to(device)) # 클래스 별 가중치(weight)를 loss에 인자로 전달
            label_loss += self.alpha*loss_funct(logits.view(-1, self.num_labels), labels.view(-1)) # hate label 로스 계산
            if(self.train_target):
                label = []
                for i in target_labels:
                    tmp = [0]*self.target_num_labels
                    for j in i:
                        if j == -1:
                            break
                        else:
                            tmp[j]=1
                    label.append(tmp)
                label = torch.tensor(label).to(device)

                label_loss +=  self.beta*focal_binary_cross_entropy(logits_target, label, self.target_num_labels)
            loss = label_loss
            
            if(self.train_att or self.train_target_att):
                loss_att=0
                n = 2 if self.train_target else 1
                if(self.train_att):
                    for i in range(self.num_sv_heads): #1개의 head를 사용함 / hate rationale loss 계산
                        attention_weights=outputs[n][self.sv_layer][:,i,0,:] #마지막 층(11)의 앞 1개의 attention head를 사용
                        loss_att +=self.lam*masked_cross_entropy(attention_weights,attention_vals,attention_mask)
                if(self.train_target_att):
                    for i in range(6,6+self.num_sv_heads): # target_rationale loss 계산. 중간 이후 num_sv_heads개의 attention을 target att 학습
                        attention_weights=outputs[n][self.sv_layer][:,i,0,:] #마지막 층(11)의 6번째 attention head를 사용
                        loss_att +=self.lam*masked_cross_entropy(attention_weights,target_att,attention_mask)
                loss += loss_att
            outputs = (loss,) + outputs
        return outputs
