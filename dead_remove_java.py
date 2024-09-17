import argparse
import javalang
import os
import json
import sys
import random
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from AST2Code import AST2Code_module
from parser.utils import remove_comments_and_docstrings
MAX_ITER=10
MAX_AUGMENTATION=1

labels_ids = {'constant':'0', 'linear':'1','logn':'2', 'quadratic':'3','cubic':'4','nlogn':'5' , 'np':'6'}

def augmentation(args):
    aug_datas=[]
    
    module=AST2Code_module(dead_code=1)
    
    codes=[]
    complexity=[]
    problems=[]
    problem_source=[]
    only_label = False
    labels = []
    
    with open(f'data/{args.file_name}') as f:
        for line in f:
            line=line.strip()
            json_obj=json.loads(line)
            codes.append(remove_comments_and_docstrings(json_obj['src'],'java'))
            try:
                complexity.append(json_obj['complexity'])
                problems.append(json_obj['problem'])
                problem_source.append(json_obj['from'])
            except:
                labels.append(json_obj['label'])
                only_label = True
    
    for idx in range(len(codes)):
        origin_code = codes[idx]
        try:
            origin_code=module.AST2Code(javalang.parse.parse(origin_code))
        except:
            pass
        if only_label:
            aug_datas.append({'label': labels[idx], 'src':origin_code})
        else:
            aug_datas.append({'src':origin_code,'complexity':complexity[idx],'problem':problems[idx],'from':problem_source[idx]})
        print(f"\r{len(aug_datas)}")
    file_name=args.file_name.rstrip('.jsonl')
    with open(os.path.join('data', f'{file_name[:-5]}dead_data.jsonl'), 'w', encoding='utf8') as f:
        for i in aug_datas: f.write(json.dumps(i) + "\n")   

if __name__=='__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--file_name', required=False, help='probablilty of augmentaion',type=str,default='test_p.jsonl')
    args = parser.parse_args()
    augmentation(args)
