from tqdm import tqdm
import torch
import numpy as np
import random
from sklearn.metrics import confusion_matrix
import argparse

from AST2Code import *
from transformers import ( RobertaTokenizer,
                          AutoTokenizer,
                          set_seed,
                          )
from models import *
import os
from torch.utils.data import  DataLoader
from utils.CodeT5utils import *
from utils.PLBart_utils import *
from utils.GraphCodeBERT_utils import *

labels_ids = {'1':0, 'n':1,'logn':2, 'n_square':3,'n_cube':4,'nlogn':5 , 'np':6}
length_items=['256','512','1024','over']
complexity_items=['constant','linear','quadratic','cubic','logn','nlogn','np']
# How many labels are we using in training.
# This is used to decide size of classification head.
n_labels = len(labels_ids)

datasets= { 'CodeBERT':CodeDataset,
            'PLBART':CodeDataset,
            'GraphCodeBERT':TextDataset,
            'CodeT5':load_classify_data,
            'CodeT5+':load_classify_data,
            'UniXcoder':CodeDataset,
            'longcoder':CodeDataset}

collate_fns={'CodeBERT':collate_fn,
            'PLBART':collate_fn,
            'GraphCodeBERT':None,
            'CodeT5':None,
            'CodeT5+':None,
            'UniXcoder':collate_fn,
            'longcoder':collate_fn
            }

tokenizers={'CodeBERT':AutoTokenizer,
            'PLBART':AutoTokenizer,
            'GraphCodeBERT':RobertaTokenizer,
            'CodeT5':RobertaTokenizer,
            'CodeT5+':RobertaTokenizer,
            'UniXcoder':AutoTokenizer,
            'longcoder':AutoTokenizer}

model_names={'CodeBERT':'microsoft/codebert-base-mlm',
            'PLBART':'uclanlp/plbart-base',
            'GraphCodeBERT':'microsoft/graphcodebert-base',
            'CodeT5':'Salesforce/codet5-base',
            'CodeT5+':'Salesforce/codet5p-220m',
            'UniXcoder':'microsoft/unixcoder-base',
            'longcoder':'microsoft/longcoder-base'}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def test(args,model,tokenizer):


    test_dataset = datasets[args.model](path=args.test_path,tokenizer=tokenizer,args=args)

    device=args.device
    print('Created `test_dataset` with %d examples!'%len(test_dataset))

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_fns[args.model])
    print('Created `eval_dataloader` with %d batches!'%len(test_dataloader))
    print('Model loaded')

    model = model.to(device)
    
    val_prediction_labels = []
    val_true_labels = []

    model.eval()

    for batch in tqdm(test_dataloader, total=len(test_dataloader)):

        if args.model in ['CodeBERT','PLBART','UniXcoder','longcoder']:
            label = batch['labels'].to(device)
        else:
            label = batch[-1].to(device)

        logits = model(batch)
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        # Convert these logits to list of predicted labels values.
        val_true_labels += label.cpu().numpy().flatten().tolist()
        val_prediction_labels += logits.argmax(axis=-1).flatten().tolist()

    conf_mat = confusion_matrix(val_true_labels, val_prediction_labels)
    acc = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)

    return round(acc*100,2)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--valid_path', required=False, help='valid file path',type=str,default='valid_p.jsonl')

    parser.add_argument('--epoch', required=False, help='number of training epoch',type=int,default=15)
    parser.add_argument('--batch', required=False, help='number of batch size',type=int,default=6)
    parser.add_argument('--model', required=False, help='selelct main model',choices=['CodeBERT','PLBART','UniXcoder','longcoder', 'CodeT5', 'CodeT5+', 'GraphCodeBERT'])

    parser.add_argument('--device', required=False, help='select device for cuda',type=str,default='cuda:0')
    parser.add_argument('--seed', required=False, type=int,default=777)

    parser.add_argument('--max_code_length', required=False, help='probablilty of augmentaion',type=int,default=512)
    parser.add_argument('--max_dataflow_length', required=False, help='probablilty of augmentaion',type=int,default=128)

    parser.add_argument('--folder_name', required=False, type=str,default='data.jsonl')
    parser.add_argument('--model_folder', required=False, type=str, default='experiments_model')
    args = parser.parse_args()

    result={'origin':[]}
    len_sum=[]
    set_seed(args)
    tokenizer = tokenizers[args.model].from_pretrained(pretrained_model_name_or_path=model_names[args.model])
    model = integrated_model(args)

    for f in range(4):
        args.fold=f
        args.test_path=f'test_{f}_fold_{args.folder_name}'
        len_sum.append(len(open(f'data/test_{f}_fold_{args.folder_name}').readlines()))
        model.load_state_dict(torch.load(f'{args.model_folder}/{f}_fold_{args.model}.pt'))
        
        result['origin'].append(test(args,model,tokenizer))
        
        for i in length_items:
            args.test_path='length_split/'+f'{i}_test_{f}_fold_{args.folder_name}'

            if i in result.keys():
                result[i].append(test(args,model,tokenizer))
            else:
                result[i]=[test(args,model,tokenizer)]

        for i in complexity_items:
            args.test_path='complexity_split/'+f'{i}_test_{f}_fold_{args.folder_name}'
            if i in result.keys():
                result[i].append(test(args,model,tokenizer))
            else:
                result[i]=[test(args,model,tokenizer)]
    #normalize
    weight=len_sum/np.sum(len_sum)
    if "result" not in os.listdir():
        os.mkdir('result')

    f = open(f'result/{args.model}_k.txt', 'w')
    for i in result.keys():
        data=f'{i} mean: {round(sum(np.multiply(result[i],weight)),2)} std : {round(np.std(result[i]),2)}'
        print(data)
        f.write(data+'\n')
    f.close()
