import argparse
import os
import json
import sys
import random
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from pythoncodeaug import ASTAug
import ast
from vulture import core
MAX_ITER=10
MAX_AUGMENTATION=1

labels_ids = {'constant':'0', 'linear':'1','logn':'2', 'quadratic':'3','cubic':'4','nlogn':'5' , 'np':'6'}

def augmentation(args):
    aug_datas=[]
    
    module=ASTAug(probs={'for2while':0.2,'ternary2if':0.2,'comp2stmt':0.05},seed=7777) #mutation=1
    
    codes=[]
    complexity=[]
    problems=[]
    problem_source=[]
    
    with open(f'data/{args.file_name}') as f:
        for line in f:
            line=line.strip()
            json_obj=json.loads(line)
            codes.append(json_obj['src'])
            complexity.append(json_obj['complexity'])
            problems.append(json_obj['problem'])
            problem_source.append(json_obj['from'])
    
    for idx in range(len(codes)):
        mutations = random.randint(1,10)
        origin_code = ast.parse(codes[idx])
        for _ in range(mutations):
            try:
                origin_code=ast.fix_missing_locations(module.visit(origin_code))
            except Exception as e:
                print(e)
                break
        else:
            aug_datas.append({'src':ast.unparse(origin_code),'complexity':complexity[idx],'problem':problems[idx],'from':problem_source[idx]})
        print(f"\r{len(aug_datas)}, {mutations}")
    while len(aug_datas)<10000:
        idx = random.randint(0,len(codes)-1)
        mutations = random.randint(1,20)
        origin_code = ast.parse(codes[idx])
        for _ in range(mutations):
            try:
                origin_code=ast.fix_missing_locations(module.visit(origin_code))
            except Exception as e:
                print(e)
                break
        else:
            aug_datas.append({'src':ast.unparse(origin_code),'complexity':complexity[idx],'problem':problems[idx],'from':problem_source[idx]})
        print(f"\r{len(aug_datas)}, {mutations}")
    file_name=args.file_name.rstrip('.jsonl')
    with open(os.path.join('data', f'{file_name}_aug_folding.jsonl'), 'w', encoding='utf8') as f:
        for i in aug_datas: f.write(json.dumps(i) + "\n")   

if __name__=='__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--file_name', required=False, help='probablilty of augmentaion',type=str,default='test_p.jsonl')
    args = parser.parse_args()
    augmentation(args)
