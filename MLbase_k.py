from utils.MLutils import feature_Extractor
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from operator import add
import json
import ast
import argparse
NUM_ITER=1

def IterModel(model,train,test):
    result_dict={}
    for i in range(NUM_ITER):
        result=TrainModel(model=model, train=train,test=test, random_state=i)
        for r in result.keys():
            if r in result_dict.keys():
                result_dict[r]=list(map(add, result_dict[r], result[r]))
            else:
                result_dict[r]=result[r]
    for r in result_dict.keys():
        result_dict[r] = [ele/NUM_ITER for ele in result_dict[r]]
    return result_dict

def TrainModel(model,train,test,random_state=12):
    if model =='SVM':
        classifier = SVC(gamma=0.3,C=0.5)
    elif model =='KNN':
        classifier = KNeighborsClassifier(n_neighbors=5)
    elif model =='Kmeans':
        classifier = KMeans(n_clusters=5, random_state=random_state)
    elif model =='DecisionTree':
        classifier = DecisionTreeClassifier(random_state=random_state, max_depth=4)
    elif model =='LogisticRegression':
        classifier = LogisticRegression(random_state=random_state, solver='lbfgs', multi_class='multinomial')
    elif model =='NaiveBayse':
        classifier = BernoulliNB()
    elif model =='RandomForest':
        classifier = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, random_state=random_state)
    elif model =='MLP':
        classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=random_state, max_iter=2000)

    else:
        print('No specific model')
        return 0,0,0

    X_train = train[0]
    y_train = train[1]

    if model =='Kmeans':
        classifier.fit(X_train)
    else:
        classifier.fit(X_train, y_train)
    acc_result={}

    for i in test.keys():
        test_data = test[i]
        scores=evaluate(test_data,classifier)
        acc_result[i]=scores

    return acc_result
def evaluate(test,classifier):
    X_test = test[0]
    y_test = test[1]
    if len(y_test)==0:
        return 0
    X_test = X_test.reshape(len(y_test),-1)
    y_predicted = classifier.predict(X_test)

    acc_score = accuracy_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted, average='weighted')

    score_base = 0
    score_expanded_2 = 0
    for val_true, val_pred in zip(y_test, y_predicted):
        score_base += 1 - abs(val_pred - val_true)/6
        score_expanded_2 += max(1 - abs(val_pred - val_true)/2, 0)
    score_base /= len(y_test)
    score_expanded_2 /= len(y_test)
    return acc_score, f1, score_base, score_expanded_2
def load_data(path):
    lines=[]
    with open(path) as f:
        for line in f:
            line=line.strip()
            lines.append(json.loads(line))
    x = []
    y = []
    for i in lines:
        complexity,code=i['label'],i['src']
        try:
            if args.lang == 'python':
                ast.parse(code)
        except:
            continue
        x.append(np.array(feature_Extractor(source=code, version=1,lang=args.lang)))
        y.append(int(complexity))
    return([np.array(x),y])

def feature_base(results,args,fold):
    dead = '_d' if args.d else ''
        
    train=load_data(f'data/train_{fold}_fold_{args.train_suffix}{dead}.jsonl')
    test=load_data(f'data/test_{fold}_fold_{args.test_suffix}{dead}.jsonl')

#   length_items=['256','512','1024','over']
#   complexity_items=['constant','linear','quadratic','cubic','logn','nlogn','np']

    test_items={'origin':test}
#   for item in length_items:
#       test_items[item]=load_data(f'data/length_split/{item}_test_{fold}_fold_{args.test_suffix}{dead}.jsonl')
        
#   for item in complexity_items:
#       test_items[item]=load_data(f'data/complexity_split/{item}_test_{fold}_fold_{args.test_suffix}{dead}.jsonl')

    #model training phase
    print(f'training model: {args.model}')
    tmp_result=IterModel(model=f'{args.model}', train=train,test=test_items)

    for t in tmp_result.keys():
        if t in results.keys():
            results[t].append(tmp_result[t])
        else:
            results[t]=[tmp_result[t]]
results={}
parser = argparse.ArgumentParser()
parser.add_argument('--d', action='store_true', help='defer lr(between submodule and transformer)')
parser.add_argument('--model', required=False, help='selelct main model',choices=['DecisionTree','SVM','RandomForest'])
parser.add_argument('--train_suffix', required=False, type=str, default='data')
parser.add_argument('--test_suffix', required=False, type=str, default='data')
parser.add_argument('--lang', required=False, type=str, default='java')
args = parser.parse_args()

for i in range(4):
    feature_base(results,args,fold=i)

for i in results.keys():
    print(f'{i} : {np.mean(results[i],axis=0)} std : {np.std(results[i],axis=0)}')

