import argparse
import sys
import math
import random
import json
import numpy as np
import pandas as pd
import os
import torch
from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList,
                          AutoConfig)
import random
import torch.optim as optim
import torch.nn.functional as F
from utils.process_utils import *
from utils.text_scores import *

def load_model(model_dir, model_dtype=torch.float16):
    config_kwargs = {"output_hidden_states": True}
    config = AutoConfig.from_pretrained(model_dir, **config_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", do_sample=False, torch_dtype=model_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True


def padding_input(input_list, input_ids=0, window_size=1):

    paddings = ['*'] * (window_size-1)
    input_pad_list = paddings + input_list + paddings

    input_masked = input_pad_list[input_ids : (input_ids+window_size)]
    ids = list(range((input_ids-window_size+1), input_ids+1))

    return input_masked, ids

def tokenizer_query(tokenizer, query, fix_control1, fix_control2, fix_control3, args):
        if args.test_model in ["llama_7b", "llama_13b", "llama_70b"]:
            system_prompt = 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.'
            input_string = f'''[INST]<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{fix_control1}{query}{fix_control2}[/INST]{fix_control3}'''
            model_inputs = tokenizer([input_string], truncation=False, return_tensors='pt').to("cuda")
        else:
            system_prompt = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.'
            input_string = f'{system_prompt}\n### USER: {fix_control1}{query}\n### ASSISTANT:'
            model_inputs = tokenizer([input_string], padding=True,truncation=False, return_tensors='pt').to("cuda")

        input_length = model_inputs['input_ids'].shape[-1]
        return model_inputs, input_length


def model_generate_batch(model, tokenizer, batch_input_ids, batch_processor, args):

    def count_padding(numbers):
        reversed_numbers = torch.flip(numbers, [0])
        count_zeros_at_end = (reversed_numbers == 0).cumsum(dim=0).max().item()
    
        return count_zeros_at_end

    batch_generated_ids = model.generate(**batch_input_ids, logits_processor=[batch_processor], max_new_tokens=args.max_new_tokens, num_beams=1, do_sample=False)
    batch_true_generated_ids = batch_generated_ids[:,batch_input_ids['input_ids'].shape[1]:]
    num_paddings_last = [count_padding(batch_true_generated_ids[i]) for i in range(batch_true_generated_ids.shape[0])]

    batch_probs = []; batch_logsum = []
    _batch_true_generated_ids = []
    num_generated_tokens = batch_true_generated_ids.shape[1]
    for i in range(batch_true_generated_ids.shape[0]):
        probs = batch_processor.prob[i][args.hand_cutting_num:(num_generated_tokens-num_paddings_last[i])]
        batch_probs.append(probs)
        batch_logsum.append(sum_of_logs(probs))
        _batch_true_generated_ids.extend([batch_true_generated_ids[i, :(num_generated_tokens-num_paddings_last[i])]])
        
    batch_processor = []
    batch_response = tokenizer.batch_decode(_batch_true_generated_ids, skip_special_tokens=True)
    
    return batch_probs, batch_logsum, batch_response, _batch_true_generated_ids


def model_embed(model, mask_input, fix_input_length, selected_ids):
    mask_input_temp = mask_input
    for i, layers in enumerate(model.model.layers):
        mask_output_temp = model.model.layers[i](mask_input_temp)
        mask_input_temp = mask_output_temp[0]
    mask_output_temp = model.model.norm(mask_input_temp)
    mask_output = model.lm_head(mask_output_temp)
    mask_output = mask_output.log_softmax(-1).to('cuda')
    mask_selected_probs = mask_output[:,:-1,:][:,torch.arange(len(selected_ids)), selected_ids]
    mask_logits = mask_selected_probs[:,(fix_input_length-1):]

    return mask_logits

def model_generate(input_ids, model, tokenizer, processor, args):
    generated_ids = model.generate(input_ids, logits_processor=[processor], max_new_tokens=args.max_new_tokens, num_beams=1, do_sample=False)
    true_generated_ids = generated_ids[:, input_ids.shape[1]:]
    probs = processor.prob[args.hand_cutting_num:]
    processor.prob = []
    logsum = sum_of_logs(probs)
    response = tokenizer.batch_decode(true_generated_ids, skip_special_tokens=True)[0]

    return probs, logsum, response


def mean(column):
    filtered = [x for x in column if x is not None]
    return sum(filtered) / len(filtered) if filtered else None

def sum_of_logs(lst):
    return math.exp(sum(math.log(x) for x in lst) / len(lst))

def save_dict2json(json_file_path, new_data):
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    data.update(new_data)
    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)