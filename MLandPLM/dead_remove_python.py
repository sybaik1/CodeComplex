import argparse
import os
import json
import sys
import random
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import ast
from vulture import core
import pyparsing
from data_processing.split import remove_comments
from parser.utils import remove_comments_and_docstrings, do_file
MAX_ITER=10
MAX_AUGMENTATION=1

labels_ids = {'constant':'0', 'linear':'1','logn':'2', 'quadratic':'3','cubic':'4','nlogn':'5' , 'np':'6'}

def augmentation(args):
    aug_datas = []

    codes=[]
    complexity=[]
    problems=[]
    problem_source=[]
    only_label = False
    labels = []
    
    with open(f'data/{args.file_name}') as f:
        for line in f:
            json_obj=json.loads(line)
            code = json_obj['src']
            commentFilter = pyparsing.pythonStyleComment.suppress()
            new_code = commentFilter.transformString(code)
            try:
                code = ast.unparse(ast.parse(new_code))
            except:
                pass
            try:
                new_code = do_file(code) 
                code = ast.unparse(ast.parse(new_code))
            except:
                pass
            # codes.append(remove_comments_and_docstrings(json_obj['src'],'python'))
            codes.append(code)
            try:
                complexity.append(json_obj['complexity'])
                problems.append(json_obj['problem'])
                problem_source.append(json_obj['from'])
            except:
                labels.append(json_obj['label'])
                only_label = True

    for idx in range(len(codes)):
        v = core.Vulture()
        origin_code = codes[idx]
        v.scan(origin_code)
        origin_code = origin_code.splitlines()
        unreached = [{item.first_lineno,item.size} for item in v.unreachable_code]
        print(v.unreachable_code)
        for line,size in unreached[::-1]:
            print(origin_code, 'deleteline:',line)
            del origin_code[line-1]
        origin_code = '\n'.join(origin_code)
        if only_label:
            aug_datas.append({'label': labels[idx], 'src':origin_code})
        else:
            aug_datas.append({'src':origin_code,'complexity':complexity[idx],'problem':problems[idx],'from':problem_source[idx]})
    file_name=args.file_name.rstrip('.jsonl')
    with open(os.path.join('data', f'{file_name[:-5]}dead_data.jsonl'), 'w', encoding='utf8') as f:
        for i in aug_datas: f.write(json.dumps(i) + "\n")   
        print(len(aug_datas))

if __name__=='__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--file_name', required=False, help='probablilty of augmentaion',type=str,default='test_p.jsonl')
    args = parser.parse_args()
    augmentation(args)
