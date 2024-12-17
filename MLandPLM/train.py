from tqdm import tqdm

import os

import torch
import numpy as np

from sklearn.metrics import confusion_matrix
import argparse


from transformers import ( RobertaTokenizer,
                          AutoTokenizer, 
                          get_linear_schedule_with_warmup,
                          set_seed)
from models import *
from torch.utils.data import  DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import random

from utils.CodeT5utils import *
from utils.PLBart_utils import *
from utils.GraphCodeBERT_utils import *
import re
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
digit=re.compile("\d+")
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def train(args):
 
    set_seed(args)
    tokenizer = tokenizers[args.model].from_pretrained(pretrained_model_name_or_path=model_names[args.model])
    
    train_dataset = datasets[args.model](path=args.train_path,tokenizer=tokenizer,args=args)
    test_dataset = datasets[args.model](path=args.test_path,tokenizer=tokenizer,args=args)

    data_type = "random" if "random" in args.train_path else f'{digit.findall(args.test_path)[0]}_fold'
    model = integrated_model(args)
    device=args.device
    # print(device)
    print('Created `train_dataset` with %d examples!'%len(train_dataset))
    print('Created `test_dataset` with %d examples!'%len(test_dataset))

    # Move pytorch dataset into dataloader.
    # random_probs parameter for augmentation. if random_probs == 0 then no augmentation.

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fns[args.model])
    print('Created `train_dataloader` with %d batches!'%len(train_dataloader))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_fns[args.model])
    print('Created `eval_dataloader` with %d batches!'%len(test_dataloader))
    datatype=args.train_path
    _model=f'data: {datatype} model:{args.model}'

    eventid = datetime.now().strftime(f'runs/{_model}-%Y_%m_%d_%H:%M')
    writer = SummaryWriter(eventid+args.u)
    args.max_steps=args.epoch*len(train_dataloader)
    args.warmup_steps=args.max_steps//5
    print('Model loaded')
    optimizer = torch.optim.AdamW(model.parameters(),
        lr = 2e-6, # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps = 1e-8, # args.adam_epsilon  - default is 1e-8.
        weight_decay=1e-2
        )

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)

    criterion = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    best_loss = np.inf
    best_acc = 0

    for epoch in range(args.epoch):

        total_loss = 0
        val_total_loss = 0

        predictions_labels = []
        true_labels = []

        val_prediction_labels = []
        val_true_labels = []

        model.train()

        for batch in tqdm(train_dataloader, total=len(train_dataloader)):

            model.zero_grad()

            if args.model in ['CodeBERT','PLBART','UniXcoder','longcoder']:
                label = batch['labels'].to(device)
            else:
                label = batch[-1].to(device)

            logits = model(batch)

            loss = criterion(logits, label)

            total_loss += loss.detach().cpu().item()

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

            # Convert these logits to list of predicted labels values.
            true_labels += label.cpu().numpy().flatten().tolist()
            predictions_labels += logits.argmax(axis=-1).flatten().tolist()
            torch.cuda.empty_cache()
        conf_mat = confusion_matrix(true_labels, predictions_labels)
        acc = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)

        writer.add_scalar('Training loss', total_loss / len(train_dataloader), epoch)
        writer.add_scalar('Training accuracy',  acc, epoch)

        print('Epoch {}, Train loss: {}, accuracy: {} %'.format(epoch, total_loss / len(train_dataloader), acc*100))

        model.eval()

        for batch in tqdm(test_dataloader, total=len(test_dataloader)):

            if args.model in ['CodeBERT','PLBART','UniXcoder','longcoder']:
                label = batch['labels'].to(device)
            else:
                label = batch[-1].to(device)

            logits = model(batch)
            loss = criterion(logits, label)

            val_total_loss += loss.detach().item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

            # Convert these logits to list of predicted labels values.
            val_true_labels += label.cpu().numpy().flatten().tolist()
            val_prediction_labels += logits.argmax(axis=-1).flatten().tolist()

        conf_mat = confusion_matrix(val_true_labels, val_prediction_labels)
        acc = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)

        writer.add_scalar('validation loss', val_total_loss / len(test_dataloader), epoch)
        writer.add_scalar('validation accuracy',  acc, epoch)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'{args.model_folder}/{data_type}_{args.model}'+'.pt')
            print(f'Best acc model saved at accuracy {acc}')
        if val_total_loss < best_loss:
            best_loss = val_total_loss
            torch.save(model.state_dict(), f'{args.model_folder}/loss_{data_type}_{args.model}'+'.pt')
            print(f'Best loss model saved at loss {val_total_loss}')
        print('Validation loss: {}, accuracy: {} %'.format(val_total_loss / len(test_dataloader), acc*100))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--u', required=False, help='unique string for tensorboard',type=str,default='')

    parser.add_argument('--train_path', required=False, help='train file path',type=str,default='train_p.jsonl')
    parser.add_argument('--test_path', required=False, help='test file path',type=str,default='test_p.jsonl')

    parser.add_argument('--epoch', required=False, help='number of training epoch',type=int,default=15)
    parser.add_argument('--batch', required=False, help='number of batch size',type=int,default=6)

    parser.add_argument('--model', required=False, help='selelct main model',choices=['CodeBERT','PLBART','GraphCodeBERT','CodeT5','CodeT5+','UniXcoder','longcoder'])

    parser.add_argument('--device', required=False, help='select device for cuda',type=str,default='cuda:0')
    parser.add_argument('--seed', required=False, type=int,default=770)

    parser.add_argument('--max_code_length', required=False, help='probablilty of augmentaion',type=int,default=512)
    parser.add_argument('--max_dataflow_length', required=False, help='probablilty of augmentaion',type=int,default=128)
    parser.add_argument('--model_folder', required=False, type=str, default='models')


    parser.add_argument('--lr', required=False,help='learning rate' ,type=float,default=2e-6)
    parser.add_argument('--eps', required=False,help='adam_epsilon' ,type=float,default=1e-8)
    parser.add_argument('--wd', required=False,help = 'weight decay', type=float,default=1e-2)
    args = parser.parse_args()
    os.makedirs(args.model_folder, exist_ok=True)
    train(args)
