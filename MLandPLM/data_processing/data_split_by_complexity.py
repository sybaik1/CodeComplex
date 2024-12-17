import argparse
import os
import json
def split(args):
    if "complexity_split" not in os.listdir("data"):
        os.mkdir('data/complexity_split')
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

    length_dict={'constant':[],'logn':[], 'linear':[],'quadratic':[],'nlogn':[],'np':[],'cubic':[]}
    for i,code in enumerate(codes):
        if complexity[i]==0:
            length_dict['constant'].append({'src':code,'label':complexity[i]})
        elif complexity[i]==1:
            length_dict['linear'].append({'src':code,'label':complexity[i]})
        elif complexity[i]==2:
            length_dict['logn'].append({'src':code,'label':complexity[i]})
        elif complexity[i]==3:
            length_dict['quadratic'].append({'src':code,'label':complexity[i]})
        elif complexity[i]==4:
            length_dict['cubic'].append({'src':code,'label':complexity[i]})
        elif complexity[i]==5:
            length_dict['nlogn'].append({'src':code,'label':complexity[i]})
        elif complexity[i]==6:
            length_dict['np'].append({'src':code,'label':complexity[i]})

    for k in length_dict.keys():
        print(k,' length :',len(length_dict[k]))
        with open(os.path.join('data/complexity_split', f'{k}_{args.file_name}'), 'w', encoding='utf8') as f:
            for i in length_dict[k]: f.write(json.dumps(i) + "\n")   

if __name__=='__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--file_name', required=False, help='split target file name',type=str,default='test_0_fold.jsonl')
    args = parser.parse_args()
    split(args)
