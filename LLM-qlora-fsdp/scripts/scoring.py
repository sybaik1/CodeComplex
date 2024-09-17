import path
import os
import argparse
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
import tqdm
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', '-m',
    type=str,
    help="The LLM used for the inference",
    required=True,
    dest='model',
)
parser.add_argument(
    '--pretrained', '-p',
    action="store_true",
)
run_args = parser.parse_args()

# script parameters
model_id = f"{run_args.model}"                          # Hugging Face model id
model_name = model_id.split('/')[-1]
dataset_path = "codecomplex-simple"                        # path to dataset
pretrained = run_args.pretrained

# output directory
if pretrained:
    response_file = f"./response/{model_name}-{dataset_path}-pretrained.txt"
    score_file = f"./score/{model_name}-{dataset_path}-pretrained.txt"
    
else:
    response_file = f"./response/{model_name}-{dataset_path}.txt"
    score_file = f"./score/{model_name}-{dataset_path}.txt"
Path('./response/').mkdir(parents=True, exist_ok=True)
Path('./score/').mkdir(parents=True, exist_ok=True)

def text2index(complexity):
    complexity = complexity.lower()
    if complexity == "logarithmic":
        return 1
    symbol = ['o(1)', 'o(log n)', 'n', 'o(n log n)', 'n^2', 'n^3']
    if complexity in symbol:
        return symbol.index(complexity)
    if complexity == "np" or complexity == "factorial":
        complexity = "exponential"
    complexity_list = ['constant', 'logn', 'linear', 'nlogn', 'quadratic', 'cubic', 'exponential']
    if complexity == "error" or complexity == "no responce" or complexity == "polynomial":
        return -1
    return complexity_list.index(complexity)


val_mapping = [0,1,2,3,4,6,8]
def scoreing_base(base, val_mapping=[0,1,2,3,4,5,6]):
    return max(1 - abs(val_mapping[val_pred] - val_mapping[val_true])/base, 0)

score_base = 0
score_expanded_2 = 0
score_expanded_3 = 0
score_base_mapping = 0
score_expanded_mapping_2 = 0
score_expanded_mapping_3 = 0
val_trues, val_preds = [], []
with open(f"{response_file}", 'r') as f:
    lines = list(json.loads(_) for _ in f)
for line in lines:
    val_true, val_pred = text2index(line["answer"].strip()), text2index(line["complexity"].strip())
    val_trues.append(val_true)
    val_preds.append(val_pred)
    if val_pred == -1:
        continue
    score_base += 1 - abs(val_pred - val_true)/6
    score_expanded_2 += max(1 - abs(val_pred - val_true)/2, 0)
    score_expanded_3 += max(1 - abs(val_pred - val_true)/3, 0)

    score_base_mapping += 1 - abs(val_mapping[val_pred] - val_mapping[val_true])/8
    score_expanded_mapping_2 += max(1 - abs(val_mapping[val_pred] - val_mapping[val_true])/2, 0)
    score_expanded_mapping_3 += max(1 - abs(val_mapping[val_pred] - val_mapping[val_true])/3, 0)
accuracy = accuracy_score(val_trues, val_preds, normalize=True)
f1 = f1_score(val_trues, val_preds, average='weighted')
score_base /= len(lines)
score_expanded_2 /= len(lines)
score_expanded_3 /= len(lines)
score_base_mapping /= len(lines)
score_expanded_mapping_2 /= len(lines)
score_expanded_mapping_3 /= len(lines)
class_acc = [accuracy_score(np.array(val_trues) == i, np.array(val_preds) == i) for i in range(7)]
print(len(val_trues))

with open(f"{score_file}", 'w') as f:
    f.write(f'{model_id}{"-pretrained" if pretrained else ""} with {dataset_path}\n')
    f.write('Scores\n')
    f.write(f'acc: {accuracy:.3f}, f1 score: {f1:.3f}\n')
    f.write(f'base: {score_base:.3f}, exp 2: {score_expanded_2:.3f}, exp 3: {score_expanded_3:.3f}\n')
    f.write('Scores with [0,1,2,3,4,6,8]\n')
    f.write(f'base: {score_base_mapping:.3f}, exp 2: {score_expanded_mapping_2:.3f}, exp 3: {score_expanded_mapping_3:.3f}\n')
    for acc in class_acc:
        f.write(f'{acc*100:.1f} &')
    f.write(f'{f1_score(val_trues, val_preds, average="micro")*100:.1f} &')
    f.write(f'{f1_score(val_trues, val_preds, average="macro")*100:.1f}\n')
