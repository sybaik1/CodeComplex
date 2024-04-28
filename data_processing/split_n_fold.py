import random
random.seed(770)
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from AST2Code import AST2Code_module
import javalang
import os
import json

labels_ids = {'constant':'0', 'linear':'1','logn':'2', 'quadratic':'3','cubic':'4','nlogn':'5' , 'np':'6'}
rev = {v: k for k, v in labels_ids.items()}
module=AST2Code_module()
import re
def delete_import(code):
    import_rule=re.compile('import{1}[^;]*;')
    package_rule=re.compile('package{1}[^;]*;')
    code=import_rule.sub('',code)
    code=package_rule.sub('',code)
    return code

def remove_comments(string, lang):
    if lang == 'java':
        pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    elif lang == 'python':
        pattern = r"(\".*?\"|\'.*?\'|=[^\S\n\t]*\"\"\".*\"\"\")|(#[^\r\n]*$|\"\"\".*\"\"\")"
    else:
        return string
    # first group captures quoted strings (double or single)
    # second group captures comments (//single-line or /* multi-line */)
    regex = re.compile(pattern, re.MULTILINE|re.DOTALL)
    def _replacer(match):
        # if the 2nd group (capturing comments) is not None,
        # it means we have captured a non-quoted (real) comment string.
        if match.group(2) is not None:
            return "" # so we will return empty to remove the comment
        else: # otherwise, we will return the 1st group
            return match.group(1) # captured quoted-string
    return regex.sub(_replacer, string)
    
def preprocessing(code,lang):
    no_comment_code = remove_comments(code,lang)
    if lang == 'java':
        # no_comment_code=delete_import(no_comment_code)
        try:
            converted_code=convert(no_comment_code)
            return converted_code
        except:
            pass
    return no_comment_code

def convert(code):
    return module.AST2Code(javalang.parse.parse(code))

def n_fold_sample(data,size,n=5):
    train={}
    test={}
    item_dict={}
    length_dict={}

    for item in data:
        if item[1] in item_dict.keys():
            item_dict[item[1]].add(item[2])
            length_dict[item[1]]+=1
        else:
            item_dict[item[1]]=set([item[2]])
            length_dict[item[1]]=1
    
    test_n_idx={}
    reset_cycle=10 # if cycle over the reset_cycle, reset all data list

    for comple in item_dict.keys():
        test_n_idx[comple]=[]
        cur_n=1
        tmp_item_dicy=item_dict[comple]
        cycle=0
        while True:
            test_idx=random.sample(tmp_item_dicy, int(len(tmp_item_dicy)*(1-size)))
            train[comple]=[]
            test[comple]=[]
            for item in data:
                if item[1]== comple:
                    if item[2] in test_idx:
                        test[comple].append([item[0],item[1]])
                    else:
                        train[comple].append([item[0],item[1]])
        
            if (len(train[comple])/length_dict[comple])<(size*1.2) and (len(train[comple])/length_dict[comple])>(size*0.8):
                cur_n+=1
                test_n_idx[comple].append(test_idx)
                tmp_item_dicy=tmp_item_dicy-set(test_idx)
            elif cycle == reset_cycle:
                print('reset')
                cur_n=1
                cycle=0
                tmp_item_dicy=item_dict[comple]
                test_n_idx[comple]=[]
            if cur_n == n:
                test_n_idx[comple].append(tmp_item_dicy)
                print(comple)
                break
            cycle+=1
    return test_n_idx

def delete_error(datas):
    
    for idx in reversed(range(len(datas))):
        source=datas[idx][0]
        try:
            tree=javalang.parse.parse(source)
            converted=module.AST2Code(tree)
            javalang.parse.parse(converted)
        except:
            del datas[idx]

def print_distribution(data,name):
    dist={}
    for t in data:
        complexity=t['label']
        if complexity in dist.keys():
            dist[complexity]+=1/len(data)
        else:
            dist[complexity]=1/len(data)
    dist=sorted(dist.items())
    tmp_dist={}
    for t in dist:
        tmp_dist[rev[str(int(t[0]))]]=t[1]
    print(f'--------{name}--------')
    for i in tmp_dist:
        print(i,":",round(tmp_dist[i],3))
        
def generate_data(data,test_n_idx):
    for fold in range(len(test_n_idx['0'])):
        train_data=[]
        test_data=[]
        for d in data:
            if d[2] in test_n_idx[d[1]][fold]:
                test_data.append({'label':d[1],"src":d[0]})
            else:
                train_data.append({'label':d[1],"src":d[0]})

        print(f'{fold}_train:',len(train_data),'\n','test:',len(test_data))
        print(f'{fold}_train:',round(len(train_data)/(len(train_data)+len(test_data)),3),'\n','test:',round(len(test_data)/(len(train_data)+len(test_data)),3))

        with open(os.path.join('data', f'train_{fold}_fold_{args.file_name}') , encoding= "utf-8",mode="w") as file: 
            for i in train_data: file.write(json.dumps(i) + "\n")
        with open(os.path.join('data', f'test_{fold}_fold_{args.file_name}') , encoding= "utf-8",mode="w") as file: 
            for i in test_data: file.write(json.dumps(i) + "\n")   
        print_distribution(train_data,'train')
        print_distribution(test_data,'test')
        print('\n\n\n')

def split(args):
    datas = []
    with open(f'data/{args.file_name}') as f:
        for line in f:
            line=line.strip()
            obj=json.loads(line)
            datas.append([preprocessing(obj['src'],args.lang), labels_ids[obj['complexity']], obj['problem']])
    prev_len=len(datas)
    if args.lang == 'java':
        delete_error(datas)
    print('delete error:',prev_len-len(datas),'out of',prev_len)
    test_n_idx=n_fold_sample(datas,args.size,n=args.fold)
    print('splited!')
    generate_data(datas,test_n_idx)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', required=False, help='file path',type=str,default='data.jsonl')
    parser.add_argument('--size', required=False, help='train-valid ratio of each split',type=float,default=0.8)
    parser.add_argument('--fold', required=False, help='number of fold',type=int,default=5)
    parser.add_argument('--lang', required=False, type=str, default='java')
    args = parser.parse_args()
    split(args)
