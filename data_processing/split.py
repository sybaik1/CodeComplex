import random
random.seed(100)
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from AST2Code import AST2Code_module
import javalang
import os
import json

labels_ids = {'constant':'0', 'linear':'1','logn':'2', 'quadratic':'3','cubic':'4','nlogn':'5' , 'np':'6'}
rev = {v: k for k, v in labels_ids.items()}
from sklearn.model_selection import train_test_split
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
    return module.AST2Code(javalange.parse.parse(code))

def delete_error(datas):
    module=AST2Code_module()
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
        complexity=t['complexity']
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

def split(args):
    datas = []

    with open(f'data/{args.file_name}', encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            obj=json.loads(line)
            datas.append([preprocessing(obj['src'],args.lang), obj['complexity'], obj['problem']])
    if args.lang == 'java':
        delete_error(datas)
    codes=[]
    complexity=[]

    for item in datas:
        codes.append([item[0],item[2]])
        complexity.append(item[1])

    x_train, x_test, y_train, y_test = train_test_split(codes, complexity, test_size=1-args.size, random_state=777, stratify=complexity)
    train=[]
    test=[]
    for idx in range(len(x_train)):
        train.append({"complexity":y_train[idx],"problem":x_train[idx][1],"src":x_train[idx][0]})
    for idx in range(len(x_test)):
        test.append({"complexity":y_test[idx],"problem":x_test[idx][1],"src":x_test[idx][0]})
    print('train:',len(train),'\n','test:',len(test))
    print('train:',round(len(train)/(len(train)+len(test)),3),'\n','test:',round(len(test)/(len(train)+len(test)),3))

    with open(os.path.join('data', f'train_random_{args.file_name}') , encoding= "utf-8",mode="w") as file: 
        for i in train: file.write(json.dumps(i) + "\n")
    with open(os.path.join('data', f'test_random_{args.file_name}') , encoding= "utf-8",mode="w") as file: 
        for i in test: file.write(json.dumps(i) + "\n")


    #print_distribution(train,'train')
    #print_distribution(test,'test')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', required=False, help='file path',type=str,default='data.jsonl') 
    parser.add_argument('--size', required=False, help='learning rate',type=float,default=0.8)
    parser.add_argument('--lang', required=False, type=str, default='java')
    args = parser.parse_args()
    split(args)
