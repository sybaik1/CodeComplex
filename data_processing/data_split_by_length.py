import argparse
import javalang
import os
import json
import tokenizer
def split(args):
    if "length_split" not in os.listdir("data"):
        os.mkdir('data/length_split')
    lines=[]
    with open(f'data/{args.file_name}') as f:
        for line in f:
            line=line.strip()
            lines.append(json.loads(line))
    codes=[]
    complexity=[]
    for line in lines:
        complexity.append(int(line['label']))
        codes.append(line['src'])
    length_dict={'256':[],'512':[], '1024':[],'over':[]}
    code_length={'256':0,'512':0, '1024':0,'over':0}

    for i,code in enumerate(codes):
        if args.lang == 'java':
            codelength = len(list(javalang.tokenizer.tokenize(code)))
        elif args.lang == 'python':
            codelength = len(list(tokenizer.tokenize(code)))

        if codelength<256:
            length_dict['256'].append({'src':code,'label':complexity[i]})
            code_length['256']+=len(code)
        elif codelength<512:
            length_dict['512'].append({'src':code,'label':complexity[i]})
            code_length['512']+=len(code)

        elif codelength<1024:
            length_dict['1024'].append({'src':code,'label':complexity[i]})
            code_length['1024']+=len(code)

        else:
            length_dict['over'].append({'src':code,'label':complexity[i]})
            code_length['over']+=len(code)

    for k in length_dict.keys():
        print(k,' length :',len(length_dict[k]))
        print(k,' mean :',code_length[k]/len(length_dict[k]),end='\n\n')

        with open(os.path.join('data/length_split', f'{k}_{args.file_name}'), 'w', encoding='utf8') as f:
            for i in length_dict[k]: f.write(json.dumps(i) + "\n")   

if __name__=='__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--file_name', required=False, help='probablilty of augmentaion',type=str,default='test_0_fold.jsonl')
    parser.add_argument('--lang', required=False, type=str, default='java')
    args = parser.parse_args()
    split(args)
