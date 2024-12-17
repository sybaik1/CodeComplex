import torch, os, multiprocessing
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed
)
from trl import SFTTrainer, SFTConfig
from peft.utils.other import fsdp_auto_wrap_policy
from accelerate import Accelerator
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', '-m',
    type=str,
    help="The LLM used for the inference",
    required=True,
    dest='model'
)
parser.add_argument(
    '--checkpoint', '-c',
    type=str,
    help="use checkpoint (True or False)",
    required=False,
    default=False,
    dest='checkpoint',
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

################
# Setup & PEFT
################
accelerator = Accelerator()
set_seed(42)

# use bf16 and FlashAttention if supported
if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
    attn_implementation = 'flash_attention_2'
else:
    compute_dtype = torch.float16
    attn_implementation = 'sdpa'

# for Gemma2 model use eager
if run_args.model.split('/')[-1].split('-')[0] == 'gemma':
    attn_implementation = 'eager'

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
    # target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"],
    modules_to_save=["lm_head", "embed_tokens"] # add if you want to use the Llama 3 instruct template
)
# script parameters
model_id = f"{run_args.model}"                          # Hugging Face model id
model_name = model_id.split('/')[-1]
dataset_path = "codecomplex-simple"                        # path to dataset
output_dir = f"./{model_name}-{dataset_path}"       # Temporary output directory for model checkpoints
save_dir = f"./trained/{model_name}-{dataset_path}" # Temporary output directory for model checkpoints
gradient_checkpointing = True
# SFT config training parameters
training_arguments = SFTConfig(
    do_eval=True,
    log_level="debug",
    dataset_text_field="text",
    packing=True,
    output_dir=output_dir,                # Temporary output directory for model checkpoints
    report_to="tensorboard",              # report metrics to tensorboard
    learning_rate=0.0002,                 # learning rate 2e-4
    lr_scheduler_type="linear",           # learning rate scheduler
    num_train_epochs=6,                   # number of training epochs
    per_device_train_batch_size=1,        # batch size per device during training
    per_device_eval_batch_size=1,         # batch size for evaluation
    gradient_accumulation_steps=2,        # number of steps before performing a backward/update pass
    optim="adamw_torch",                  # use torch adamw optimizer
    logging_steps=10,                     # log every 10 steps
    save_strategy="epoch",                # save checkpoint every epoch
    eval_strategy="epoch",                # evaluate every epoch
    max_grad_norm=0.3,                    # max gradient norm
    warmup_ratio=0.03,                    # warmup ratio
    bf16=True,                            # use bfloat16 precision
    max_seq_length=2048,                  # max sequence length for model and packing of the dataset
    ddp_find_unused_parameters=False,
    #   # use for Instructed version else comment out
    #   dataset_kwargs={
    #       "add_special_tokens": False,  # We template with special tokens
    #       "append_concat_token": False,  # No need to add additional separator token
    #   },
)

################
# Dataset
################
train_dataset = load_dataset(
    "json",
    data_files=os.path.join(dataset_path, "train_dataset.json"),
    split="train",
)
test_dataset = load_dataset(
    "json",
    data_files=os.path.join(dataset_path, "test_dataset.json"),
    split="train",
)

################
# Model & Tokenizer
################
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir='./.cache/huggingface/hub/',
    trust_remote_code=True,
    use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# template dataset
def template_dataset(examples):
    return{"text":  tokenizer.apply_chat_template(examples["messages"], tokenize=False)}
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


if 'Llama' in model_id:
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
elif 'gemma' in model_id.lower() or 'phi' in model_id.lower():
    tokenizer.padding_side = 'right'
    train_dataset = train_dataset.map(change_user)
    test_dataset = test_dataset.map(change_user)

train_dataset = train_dataset.map(template_dataset, remove_columns=["messages"])
test_dataset = test_dataset.map(template_dataset, remove_columns=["messages"])

# print random sample
# with training_args.main_process_first(
#     desc="Log a few random samples from the processed training set"
# ):
#     for index in random.sample(range(len(train_dataset)), 2):
#         print(train_dataset[index]["text"])

# Model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_quant_storage=compute_dtype,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
#    quantization_config=bnb_config,
    attn_implementation=attn_implementation,
    torch_dtype=compute_dtype,
    cache_dir='./.cache/huggingface/hub/',
    use_cache=False if gradient_checkpointing else True,  # this is needed for gradient checkpointing
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

for name, param in model.named_parameters():
    # freeze base model's layers
    param.requires_grad = False
def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)


model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})

################
# Training
################
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)
if trainer.accelerator.is_main_process:
    trainer.model.print_trainable_parameters()

# from transformers.trainer import _is_peft_model
# print(_is_peft_model(trainer.model))

# fsdp_plugin = trainer.accelerator.state.fsdp_plugin
# fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

##########################
# Train model
##########################
trainer.train(
    resume_from_checkpoint=run_args.checkpoint)

##########################
# SAVE MODEL FOR SAGEMAKER
##########################
if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
trainer.save_model(output_dir)

# ## MERGE PEFT AND BASE MODEL ####
# from peft import AutoPeftModelForCausalLM
 
# # Load PEFT model on CPU
# model = AutoPeftModelForCausalLM.from_pretrained(
#     output_dir,
#     torch_dtype=torch.float16,
#     low_cpu_mem_usage=True,
#     cache_dir='./.cache/huggingface/hub/',
# )
# # Merge LoRA and base model and save
# merged_model = model.merge_and_unload()
# merged_model.save_pretrained(save_dir,safe_serialization=True, max_shard_size="5GB")
