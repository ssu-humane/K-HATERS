from transformers.models.bert.modeling_bert import *
from torch import nn
from .utils import *
from transformers import AutoModel

class SC_weighted_BERT(BertPreTrainedModel):
    def __init__(self, config,params):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.target_num_labels = params['target_num_classes']
        self.alpha = params['hate_alpha']
        self.beta = params['target_beta']
        self.model_path = params['path_files']
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier_target = nn.Linear(config.hidden_size, params['target_num_classes'])
        self.init_weights() 
 
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
        logits_target = self.classifier_target(pooled_output) # target classification
        outputs = (logits, logits_target) + outputs[2:]
            
        if labels is not None:
            label_loss = 0
            loss_funct = CrossEntropyLoss()
            label_loss += self.alpha*loss_funct(logits.view(-1, self.num_labels), labels.view(-1))
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
            
            outputs = (loss,) + outputs
        return outputs
