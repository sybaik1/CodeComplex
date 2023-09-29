import torch
import torch.nn as nn
from transformers import (AutoConfig, RobertaTokenizer, RobertaForSequenceClassification,AutoModelForSequenceClassification,
                          AutoModel,T5Config,T5ForConditionalGeneration,RobertaConfig,
                          set_seed,
                          )
import torch.nn.functional as F
class CodeT5ClassifierModel(nn.Module):
    def __init__(self, encoder, config, tokenizer,max_source_length,num_class):
        super(CodeT5ClassifierModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classification=nn.Linear(config.hidden_size, num_class)
        self.max_source_length=max_source_length

    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def forward(self, source_ids=None, labels=None):
        source_ids = source_ids.view(-1, self.max_source_length)
        vec = self.get_t5_vec(source_ids)
        logit=self.classification(vec)
        return vec,logit

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.num_labels) 
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return x
        
class GraphCodeBERT_model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(GraphCodeBERT_model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args
    
        
    def forward(self, inputs_ids,position_idx,attn_mask,labels=None): 
        if len(inputs_ids.size())==1:#for comple model
            inputs_ids=inputs_ids.unsqueeze(0)
            position_idx=position_idx.unsqueeze(0)
            attn_mask=attn_mask.unsqueeze(0)
            
        #embedding
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        inputs_embeddings=self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
        
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx,token_type_ids=position_idx.eq(-1).long())[0]
        logits=self.classifier(outputs)

        return outputs,logits

class integrated_model(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        set_seed(1234)
        self.labels_ids = {'1':0, 'n':1,'logn':2, 'n_square':3,'n_cube':4,'nlogn':5 , 'np':6}

        # for tokenizer
        self.generators={
            'CodeBERT':self.generate_CodeBERT,
            'PLBART':self.generate_PLBART,
            'CodeT5':self.generate_CodeT5,
            'CodeT5+':self.generate_CodeT5p,
            'GraphCodeBERT':self.generate_GraphCodeBERT,
            'UniXcoder':self.generate_UniXcoder,
            'longcoder':self.generate_longcoder,
            }
        self.model_name=args.model
        self.get_embeddings={
                'CodeBERT':self.get_CodeBERT_embedding,
                'PLBART':self.get_PLBART_embedding,
                'CodeT5':self.get_CodeT5_embedding,
                'CodeT5+':self.get_CodeT5_embedding,
                'GraphCodeBERT':self.get_GraphCodeBERT_embedding,
                'UniXcoder':self.get_UniXcoder_embedding,
                'longcoder':self.get_longcoder_embedding
            }
        self.pretrain = getattr(args, 'pretrain', False)
        self.generators[self.model_name]()
        self.device = args.device
    def forward(self, x):
        logit=self.get_embeddings[self.model_name](x)
        return logit
    def generate_CodeBERT(self):
        
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path='microsoft/codebert-base-mlm', num_labels=len(self.labels_ids))
        self.CodeBERT = AutoModel.from_pretrained(pretrained_model_name_or_path='microsoft/codebert-base-mlm', config=model_config)
        self.decoder = nn.Linear(model_config.hidden_size, len(self.labels_ids))

    def generate_PLBART(self):
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path='uclanlp/plbart-base', num_labels=len(self.labels_ids))
        self.PLBART = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path='uclanlp/plbart-base', config=model_config)

    def generate_CodeT5(self):
        config=T5Config.from_pretrained('Salesforce/codet5-base')
        tokenizer=RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
        encoder=T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
        self.CodeT5 = CodeT5ClassifierModel(encoder, config, tokenizer,512, len(self.labels_ids))

    def generate_CodeT5p(self):
        config=T5Config.from_pretrained('Salesforce/codet5p-220m')
        tokenizer=RobertaTokenizer.from_pretrained('Salesforce/codet5p-220m')
        encoder=T5ForConditionalGeneration.from_pretrained('Salesforce/codet5p-220m')
        self.CodeT5 = CodeT5ClassifierModel(encoder, config, tokenizer,512, len(self.labels_ids))
        
    def generate_GraphCodeBERT(self):
        model_path="microsoft/graphcodebert-base"
        config = RobertaConfig.from_pretrained(model_path,num_labels=7)
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        encoder = RobertaForSequenceClassification.from_pretrained(model_path,config=config)    
        
        if self.pretrain:
            self.GraphCodeBERT=torch.load('experiments_model/GraphCodeBERT_encoder.pt')
        else:
            self.GraphCodeBERT=GraphCodeBERT_model(encoder,config,tokenizer,None)
    def generate_UniXcoder(self):
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path='microsoft/unixcoder-base', num_labels=len(self.labels_ids))
        self.UniXcoder = AutoModel.from_pretrained(pretrained_model_name_or_path='microsoft/unixcoder-base', config=model_config)
        self.decoder = nn.Linear(model_config.hidden_size, len(self.labels_ids))
        
    def generate_longcoder(self):
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path='microsoft/longcoder-base', num_labels=len(self.labels_ids))
        self.longcoder = AutoModel.from_pretrained(pretrained_model_name_or_path='microsoft/longcoder-base', config=model_config)
        self.decoder = nn.Linear(model_config.hidden_size, len(self.labels_ids))

    def get_CodeBERT_embedding(self,x,embedding=False):
        if 'labels' in x.keys():
            del x['labels']
        if 'idx' in x.keys():
            del x['idx']
            
        output=self.CodeBERT(**x.to(self.device))

        if embedding:
            return output['pooler_output']
        else:
            return self.decoder(output['pooler_output'])
    def get_UniXcoder_embedding(self,x,embedding=False):
        if 'labels' in x.keys():
            del x['labels']
        if 'idx' in x.keys():
            del x['idx']
            
        output=self.UniXcoder(**x.to(self.device))

        if embedding:
            return output['pooler_output']
        else:
            return self.decoder(output['pooler_output'])
    def get_longcoder_embedding(self,x,embedding=False):
        if 'labels' in x.keys():
            del x['labels']
        if 'idx' in x.keys():
            del x['idx']
            
        output=self.longcoder(**x.to(self.device))

        if embedding:
            return output['pooler_output']
        else:
            return self.decoder(output['pooler_output'])
    def get_PLBART_embedding(self,x,embedding=False):
        if 'labels' in x.keys():
            del x['labels']
        output=self.PLBART(**x.to(self.device))

        if embedding:
            return output['last_hidden_state'][:,-1,:]
        else:
            return output['logits']

    def get_CodeT5_embedding(self,x,embedding=False):
        output=self.CodeT5(x[0].to(self.device))
        if embedding:
            return output[0]
        else:
            return output[1]

    def get_GraphCodeBERT_embedding(self,x,embedding=False):
        (inputs_ids,position_idx,attn_mask,
        labels)=[item.to(self.device)  for item in x]

        output=self.GraphCodeBERT(inputs_ids,position_idx,attn_mask,labels)
        if embedding:
            return output[0][:, 0, :]
        else:
            return output[1]
