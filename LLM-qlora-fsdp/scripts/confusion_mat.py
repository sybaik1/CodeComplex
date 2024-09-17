import seaborn as sns
import numpy as np
import path
import os
import argparse
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
import tqdm
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

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
    save_file = f"./confusion_matrix/{model_name}-{dataset_path}-pretrained.pdf"
else:
    response_file = f"./response/{model_name}-{dataset_path}.txt"
    score_file = f"./score/{model_name}-{dataset_path}.txt"
    save_file = f"./confusion_matrix/{model_name}-{dataset_path}.pdf"
Path('./confusion_matrix/').mkdir(parents=True, exist_ok=True)


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
        return 7
    return complexity_list.index(complexity)

val_trues, val_preds = [], []
with open(f"{response_file}", 'r') as f:
    lines = list(json.loads(_) for _ in f)
for line in lines:
    val_true, val_pred = text2index(line["answer"].strip()), text2index(line["complexity"].strip())
    val_trues.append(val_true)
    val_preds.append(val_pred)

conf_mat = confusion_matrix(val_trues, val_preds)
sns.set(font_scale=1)
ax=sns.heatmap(conf_mat, annot=True,cbar=True, fmt='g')# ,cmap='Blues'
ax.set_xlabel('Prediction')
ax.set_ylabel('Ground Truth')
## Ticket labels - List must be in alphabetical order
if 7 in val_preds:
    labels = ['1', r'$\ln n$','n', r'$n \ln n$',r'$n^{2}$',r'$n^{3}$',r'$NP$-$h$', r'$ERR$']
else:
    labels = ['1', r'$\ln n$','n', r'$n \ln n$',r'$n^{2}$',r'$n^{3}$',r'$NP$-$h$']
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)
plt.savefig(save_file,dpi=300,bbox_inches='tight')
