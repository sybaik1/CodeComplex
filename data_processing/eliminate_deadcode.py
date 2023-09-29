import os
import sys
import javalang
import json
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from AST2Code import AST2Code_module
from pythoncodeaug import PythonCodeAug
import argparse

def augmentation(path):
    
    datas=[]
    with open(f'{path}.txt') as f:
        for line in f:
            line=line.strip()
            datas.append(json.loads(line))
    codes=[]
    complexity=[]
    for item in datas:
        complexity.append(item['label'])
        codes.append(item['src'])

    new_datas=[]
    fail=0
    for i,code in enumerate(codes):
        module = AST2Code_module(dead_code=1)
        try:
            tree=javalang.parse.parse(code)
            augmented_code=module.AST2Code(tree)
            new_datas.append({'label':complexity[i],'src':augmented_code})
        except:
            fail+=1
            new_datas.append({'label':complexity[i],'src':code})
    print(f'data:{path} fail:{fail}')
    with open(f'{path}_d.jsonl' , encoding= "utf-8",mode="w") as file: 
        for i in new_datas: 
            file.write(json.dumps(i) + "\n")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', required=False, type=str, default='data.jsonl')
    parser.add_argument('--lang', required=False, type=str, default='java')
    args = parser.parse_args()

    length_items=['256','512','1024','over']
    complexity_items=['constant','linear','quadratic','cubic','logn','nlogn','np']

    for f in range(4):
        path=f'data/test_{f}_fold'
        augmentation(path)
        path=f'data/train_{f}_fold'
        augmentation(path)
        # for i in length_items:
        #     path='data/length_split/'+f'{i}_test_{f}_fold'
        #     augmentation(path)
        # for i in complexity_items:
        #     path='data/complexity_split/'+f'{i}_test_{f}_fold'
        #     augmentation(path)
