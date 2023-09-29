import argparse
import javalang
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from AST2Code import AST2Code_module

def augmentation(args):
    datas=[]
    
    with open(f'data/{args.file_num}.jsonl') as f:
        for line in f:
            line=line.strip()
            json_obj=json.loads(line)
            datas.append({'src':json_obj['src'],'label':int(json_obj['label'])})
    prev_len=len(datas)

    with open(f'data/{args.extend_file}') as f:
        for line in f:
            line=line.strip()
            json_obj=json.loads(line)
            datas.append({'src':json_obj['src'],'label':int(json_obj['label'])})
            
    print(f'before : {prev_len}  | after : {len(datas)}')
    with open(os.path.join('data', f'args.file_num_extend.jsonl'), 'w', encoding='utf8') as f:
        for i in datas: f.write(json.dumps(i) + "\n")   

if __name__=='__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--file_name', required=False, help='original file path',type=int,default='train_0_fold')
    parser.add_argument('--extend_file', required=False, help='extra dataset for extend',type=int,default='extend')
    args = parser.parse_args()
    augmentation(args)