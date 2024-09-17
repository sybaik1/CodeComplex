import torch, os, multiprocessing
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
    pipeline
)
from transformers.pipelines.pt_utils import KeyDataset
import argparse
from random import randint
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

# Anthropic/Vicuna like template without the need for special tokens
LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'user' %}"
            "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)

# use bf16 and FlashAttention if supported
if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
    attn_implementation = 'flash_attention_2'
else:
    compute_dtype = torch.float16
    attn_implementation = 'sdpa'

# script parameters
model_id = f"{run_args.model}"                          # Hugging Face model id
pretrained = run_args.pretrained
model_name = model_id.split('/')[-1]
dataset_path = "codecomplex-simple"                        # path to dataset
output_dir = f"./{model_name}-{dataset_path}"       # Temporary output directory for model checkpoints
save_dir = f"./trained/{model_name}-{dataset_path}" # Temporary output directory for model checkpoints
max_length = 2048

# base model or trained model
if pretrained:
    use_model = output_dir
else:
    use_model = model_id

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    use_model,
    cache_dir='./.cache/huggingface/hub/',
    use_fast=True,
    truncation=True,
    max_length=max_length,
    max_position_embeddings=max_length,
)
tokenizer.pad_token = tokenizer.eos_token

# template dataset
def template_dataset(examples):
    return{"text":  tokenizer.apply_chat_template(examples["messages"], tokenize=False)}
def template_dataset_with_truncate(examples):
    return{"text":  tokenizer.apply_chat_template(examples["messages"], max_position_embeddings=max_length, tokenize=False)}
# preprocessing
def remove_assistant(example):
    new_messages = [
        message 
        for message in example["messages"] 
        if message["role"] != "assistant"
    ]
    example["messages"] = new_messages
    return example
    
def change_user(example):
    new_example = {"messages": []}
    new_content = ""
    for message in example["messages"]:
        if message["role"] in ["system", "user"]:
            new_content += message["content"]
        else:
            new_example["messages"].append({"role": "user", "content": new_content})
            new_example["messages"].append(message)
            new_content = ""
    if new_content != "":
        new_example["messages"].append({"role": "user", "content": new_content})
    return new_example

def truncate_text(example):
    truncated_messages = [
        message[:max_length] 
        if len(message) > max_length 
        else message 
        for message in example["messages"]
    ]
    example["messages"] = truncated_messages
    return example

# test dataset
test_dataset = load_dataset(
    "json",
    data_files=os.path.join(dataset_path, "test_dataset.json"),
    split="train",
)
test_dataset = test_dataset.map(remove_assistant)
if 'Llama' in model_id:
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
elif 'gemma' in model_id:
    tokenizer.padding_side = 'right'
    test_dataset = test_dataset.map(change_user)

if 'gemma' in model_id:
    test_dataset = test_dataset.map(template_dataset_with_truncate)
    eval_dataset = test_dataset.map(truncate_text)
else:
    eval_dataset = test_dataset.map(template_dataset)

# load model with PERT adapter
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_quant_storage=compute_dtype,
)
model = AutoModelForCausalLM.from_pretrained(
    use_model,
    quantization_config=bnb_config,
    torch_dtype=compute_dtype,
    cache_dir='./.cache/huggingface/hub/',
    device_map="auto",
    low_cpu_mem_usage=True,
)

# Test on sample
pipe = pipeline(
    'text-generation',
    model=model,
    tokenizer = tokenizer,
    torch_dtype=compute_dtype, 
    device_map="auto",
    framework="pt",
)
pipe.final_offload_hook = True
print(pipe._preprocess_params)

# Tokenizer params
tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':max_length}

# make output directory
if pretrained:
    response_path = f"./response/{model_name}-{dataset_path}-pretrained/"
else:
    response_path = f"./response/{model_name}-{dataset_path}/"
Path(response_path).mkdir(parents=True, exist_ok=True)

# inference on model
from tqdm import tqdm
responses = []
for i, data in enumerate(tqdm(KeyDataset(eval_dataset, "text"))):
    outputs = pipe(data, max_new_tokens=256, return_full_text=True, **tokenizer_kwargs)
    responses.append(outputs)
    with open(f"{response_path}/responce_{i:>04d}.txt", 'w') as f:
        f.write(repr(outputs))
