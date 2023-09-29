import javalang
from tqdm import tqdm
import os
import torch
import numpy as np

from sklearn.metrics import confusion_matrix
import argparse

from AST2Code import *
from transformers import (RobertaTokenizer,AutoTokenizer,set_seed)
from models import *
from torch.utils.data import  DataLoader

from utils.CodeT5utils import *
from utils.PLBart_utils import *
from utils.GraphCodeBERT_utils import *

labels_ids = {'1':0, 'n':1,'logn':2, 'n_square':3,'n_cube':4,'nlogn':5 , 'np':6}
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

def evaluate(args):
 
    set_seed(args)
    tokenizer = tokenizers[args.model].from_pretrained(pretrained_model_name_or_path=model_names[args.model])
    test_dataset = datasets[args.model](path=args.valid_path,tokenizer=tokenizer,args=args)

    model = integrated_model(args)

    model.load_state_dict(torch.load(f'{args.model_folder}/random_{args.model}.pt'))
    device=args.device
    print('Created `test_dataset` with %d examples!'%len(test_dataset))

    # Move pytorch dataset into dataloader.
    # random_probs parameter for augmentation. if random_probs == 0 then no augmentation.
    valid_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_fns[args.model])
    print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))
    print('Model loaded')

    model = model.to(device)
    
    val_prediction_labels = []
    val_true_labels = []

    model.eval()

    for batch in tqdm(valid_dataloader, total=len(valid_dataloader)):

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

    
    return 'accuracy: {} %'.format( round(acc*100,2))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--valid_path', required=False, help='valid file path',type=str,default='valid_random.jsonl')
    parser.add_argument('--model', required=False, help='selelct main model',choices=['CodeBERT','PLBART','UniXcoder','longcoder', 'CodeT5', 'CodeT5+','GraphCodeBERT'])
    parser.add_argument('--batch', required=False, help='number of batch size',type=int,default=6)

    parser.add_argument('--device', required=False, help='select device for cuda',type=str,default='cuda:0')
    parser.add_argument('--seed', required=False, type=int,default=777)
    parser.add_argument('--max_code_length', required=False, help='probablilty of augmentaion',type=int,default=512)
    parser.add_argument('--max_dataflow_length', required=False, help='probablilty of augmentaion',type=int,default=128)
    parser.add_argument('--file_name', required=False, type=str, default='data')
    parser.add_argument('--model_folder', required=False,type=str, default='experiments_model')
    args = parser.parse_args()
    result_val = evaluate(args)
    print(result_val)
    if "result" not in os.listdir():
        os.mkdir('result')

    f = open(f'result/{args.model}_random.txt', 'w')
    f.write(result_val+'\n')
    f.close()
