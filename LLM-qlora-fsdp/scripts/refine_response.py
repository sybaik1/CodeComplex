import path
import os
import argparse
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
import tqdm
import json
from pathlib import Path

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
    response_path = f"./response/{model_name}-{dataset_path}-pretrained/"
else:
    response_path = f"./response/{model_name}-{dataset_path}/"
Path(response_path).mkdir(parents=True, exist_ok=True)

# test dataset
test_dataset = load_dataset(
    "json",
    data_files=os.path.join(dataset_path, "test_dataset.json"),
    split="train",
)

def answer_assistant(example):
    for message in example["messages"]:
        if message["role"] == "assistant":
            answer = message
    example["answer"] = answer
    return example
test_dataset = test_dataset.map(answer_assistant)

def get_prediction(response):
    bigO_text = ['exponential', 'cubic', 'quadratic', 'nlogn', 'linear', 'logn', 'constant']
    bigO = {'np': 'exponential', '2^n': 'exponential', 'n^3': 'cubic', 'n^2': 'quadratic', 'n log n': 'nlogn', 'o(n)': 'linear', 'log n': 'logn', 'o(1)': 'constant'}

    if 'gemma' in model_id:
        assistant_text = response.split(r'<end_of_turn>')[-1].lower()
    elif 'Llama' in model_id:
        assistant_text = response.split(r'<|eot_id|>')[-1].lower()
    elif 'Mistral' in model_id:
        assistant_text = response.split(r'[/INST]')[-1].lower()
    elif 'Qwen' in model_id:
        assistant_text = response.split(r'<|im_end|>')[-1].lower()
    else:
        raise NotImplementedError
    try:
        if 'overall time' in assistant_text:
            complexity_text = assistant_text.split('overall time')[1]
        else:
            complexity_text = assistant_text.split("complexity")[1]
        try:
            comp_from_text = next(complexity for complexity in bigO_text if complexity in complexity_text)
        except StopIteration:
            comp_from_text = "ERROR"
        try:
            comp_from_symbol = next(val for key,val in bigO.items() if key in complexity_text)
        except StopIteration:
            comp_from_symbol = "ERROR"
        if comp_from_text == "ERROR" and comp_from_symbol == "ERROR":
            comp = "ERROR"
        elif comp_from_text == "ERROR":
            comp = comp_from_symbol
        elif comp_from_symbol == "ERROR":
            comp = comp_from_text
        else:
            if bigO_text.index(comp_from_text) < list(bigO.values()).index(comp_from_symbol):
                comp = comp_from_text
            else:
                comp = comp_from_symbol
    except IndexError:
        comp = "NO RESPONCE"
    return comp, assistant_text

responses = []
for i, data in enumerate(KeyDataset(test_dataset, "answer")):
    with open(f"{response_path}/responce_{i:>04d}.txt", 'r') as f:
        response = eval(f.readline())[0]['generated_text']
        comp, response = get_prediction(response)
        responses.append({"answer": data["content"].split(':')[1], "complexity": comp, "responce": response})

with open(f"{response_path[:-1]}.txt", 'w') as f:
    for response in responses:
        f.write(json.dumps(response) + '\n')
