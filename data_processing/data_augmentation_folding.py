import argparse
import javalang
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from AST2Code_folding import AST2Code_folding_module
MAX_ITER=10
MAX_AUGMENTATION=1
def augmentation(args):
    aug_datas=[]
    
    module=AST2Code_folding_module(mutation=1)
    
    codes=[]
    complexity=[]
    
    with open(f'data/{args.file_name}') as f:
        for line in f:
            line=line.strip()
            json_obj=json.loads(line)
            complexity.append(int(json_obj['label']))
            codes.append(json_obj['src'])
    
    for idx in range(len(codes)):
        try:
            origin_code=module.AST2Code(javalang.parse.parse(codes[idx]))
        except:
            pass
        aug_datas.append({'src':origin_code,'label':complexity[idx]})
    file_name=args.file_name.rstrip('.jsonl')
    with open(os.path.join('data', f'{file_name}_aug_folding.jsonl'), 'w', encoding='utf8') as f:
        for i in aug_datas: f.write(json.dumps(i) + "\n")   

if __name__=='__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--file_name', required=False, help='probablilty of augmentaion',type=str,default='test_p.jsonl')
    args = parser.parse_args()
    augmentation(args)