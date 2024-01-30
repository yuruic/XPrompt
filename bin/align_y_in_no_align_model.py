from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import argparse
import sys
import math
import random
import json
import shutil
import time
import gc
import os
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

import os


import collections
from math import sqrt

import scipy.stats

import torch
from torch import Tensor
from tokenizers import Tokenizer
from transformers import LogitsProcessor

from nltk.util import ngrams

from transformers import LogitsProcessor

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList)

from logitsProcessors import SavingLogitsProcessor, ChangingSavingLogitsProcessor

pos_list = [1, 10, 17, 19, 20, 21, 29, 31, 33, 40, 43, 45, 48, 51, 53]
neg_list = [2, 5, 7, 9, 11, 12, 14, 16, 24, 26, 27, 34, 36, 38, 39]

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def build_chat_input(query, tokenizer):
    input_string = f'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\'s questions.\n### Human: {query}\n### Assistant:'
    model_inputs = tokenizer([input_string], padding=True, truncation=False, return_tensors='pt').to("cuda")
    # print(model_inputs['input_ids'].shape)
    input_length = model_inputs['input_ids'].shape[-1]
    return model_inputs, input_length

def sum_of_logs(lst):
    return sum(math.log(x) for x in lst)


def main():
    # loading data
    with open("/data/bochuan/llm-attacks/experiments/results/individual_behavior_controls.json", 'r') as f:
        log = json.load(f)
    pos_dict = {'goal':[], 'controls':[]} # positive
    neg_dict = {'goal':[], 'controls':[]} # negative
    for i in range(len(log["goal"])):
        if i in pos_list:
            pos_dict['goal'].append(log['goal'][i])
            pos_dict['controls'].append(log['controls'][i])
        elif i in neg_list:
            neg_dict['goal'].append(log['goal'][i])
            neg_dict['controls'].append(log['controls'][i])

    # loading model
    align_model_dir = "/data/bochuan/llm-attacks/DIR/vicuna/vicuna-7b-v1.3"
    align_model = AutoModelForCausalLM.from_pretrained(align_model_dir, device_map="auto")
    align_tokenizer = AutoTokenizer.from_pretrained(align_model_dir, trust_remote_code=True, use_fast=False)
    align_tokenizer.padding_side = 'left'

    no_align_model_dir = "/data/bochuan/llm-attacks/DIR/Wizard-Vicuna-7B-Uncensored"
    no_align_model = AutoModelForCausalLM.from_pretrained(no_align_model_dir, device_map="auto")
    no_align_tokenizer = AutoTokenizer.from_pretrained(no_align_model_dir, trust_remote_code=True, use_fast=False)
    no_align_tokenizer.padding_side = 'left'

    # positive
    positive_scores = []

    all_results = {"positive":{"prob":{}, "string":{}}, "negative":{"prob":{}, "string":{}}}

    for i in range(len(pos_dict["goal"])):

        # get aligned y
        atk_string = f'{pos_dict["goal"][i]} {pos_dict["controls"][i]}'
        model_inputs, input_length = build_chat_input(atk_string, align_tokenizer)
        atk_generated_ids = align_model.generate(**model_inputs, max_new_tokens=200)
        # global true_atk_generated_ids
        align_generated_ids = atk_generated_ids[:, input_length:]
        response = align_tokenizer.batch_decode(align_generated_ids, skip_special_tokens=True)[0]
        align_generated_ids = no_align_tokenizer([response], padding=True, truncation=False, return_tensors='pt').to("cuda")
        align_generated_ids = align_generated_ids['input_ids']

        # attacking
        atk_string = f'{pos_dict["goal"][i]} {pos_dict["controls"][i]}'
        model_inputs, input_length = build_chat_input(atk_string, no_align_tokenizer)
        processor = ChangingSavingLogitsProcessor(aim=align_generated_ids[0])
        atk_generated_ids = no_align_model.generate(**model_inputs, logits_processor=[processor], max_new_tokens=200)
        # global true_atk_generated_ids
        # true_atk_generated_ids = atk_generated_ids[:, input_length:]

        # response = no_align_tokenizer.batch_decode(atk_generated_ids, skip_special_tokens=True)[0]
        # print(len(true_atk_generated_ids))
        print(f"====================================== positive case {i} =============================================")
        print(f"------------------------------- with adv prompt -------------------------------------")
        # print(response)

        attacking_scores = processor.prob
        processor.prob = []

        print(f"attacking score: {attacking_scores}")
        atk_log_sum = sum_of_logs(attacking_scores)
        atk_log_sum_drop0 = sum_of_logs(attacking_scores[1:])
        print(f"attacking score log sum: {atk_log_sum}")
        # print(f"attacking score log sum drop value which index=0: {atk_log_sum_drop0}")

        # no attacking
        print(f"------------------------------- without adv prompt -------------------------------------")
        no_atk_string = f'{pos_dict["goal"][i]}'
        model_inputs, input_length = build_chat_input(no_atk_string, no_align_tokenizer)
        processor = ChangingSavingLogitsProcessor(aim=align_generated_ids[0])
        no_atk_generated_ids = no_align_model.generate(**model_inputs, logits_processor=[processor], max_new_tokens=200)
        # global true_no_atk_generated_ids
        # true_no_atk_generated_ids = no_atk_generated_ids[:, input_length:]
        # response = no_align_tokenizer.batch_decode(no_atk_generated_ids, skip_special_tokens=True)[0]
        # print(response)
        no_attacking_scores = processor.prob
        processor.prob = []

        print(f"no attacking score: {no_attacking_scores}")
        no_atk_log_sum = sum_of_logs(no_attacking_scores)
        # no_atk_log_sum_drop0 = sum_of_logs(no_attacking_scores[1:])
        print(f"no attacking score log sum: {no_atk_log_sum}")
        positive_scores.append([atk_log_sum, no_atk_log_sum, atk_log_sum - no_atk_log_sum])
        # print(f"no attacking score log sum drop value which index=0: {no_atk_log_sum_drop0}")


    # negative
    negative_scores = []
    for i in range(len(neg_dict["goal"])):
        atk_string = f'{pos_dict["goal"][i]} {pos_dict["controls"][i]}'
        model_inputs, input_length = build_chat_input(atk_string, align_tokenizer)
        atk_generated_ids = align_model.generate(**model_inputs, max_new_tokens=200)
        # global true_atk_generated_ids
        align_generated_ids = atk_generated_ids[:, input_length:]
        response = align_tokenizer.batch_decode(align_generated_ids, skip_special_tokens=True)[0]
        align_generated_ids = no_align_tokenizer([response], padding=True, truncation=False, return_tensors='pt').to("cuda")
        align_generated_ids = align_generated_ids['input_ids']

        # attacking
        atk_string = f'{neg_dict["goal"][i]} {neg_dict["controls"][i]}'
        model_inputs, input_length = build_chat_input(atk_string, no_align_tokenizer)
        processor = ChangingSavingLogitsProcessor(aim=align_generated_ids[0])
        atk_generated_ids = no_align_model.generate(**model_inputs, logits_processor=[processor], max_new_tokens=200)
        # global true_atk_generated_ids
        # true_atk_generated_ids = atk_generated_ids[:, input_length:]

        # response = no_align_tokenizer.batch_decode(atk_generated_ids, skip_special_tokens=True)[0]
        # print(len(true_atk_generated_ids))
        print(f"====================================== negative case {i} =============================================")
        print(f"------------------------------- with adv prompt -------------------------------------")
        # print(response)

        attacking_scores = processor.prob
        processor.prob = []

        print(f"attacking score: {attacking_scores}")
        atk_log_sum = sum_of_logs(attacking_scores)
        atk_log_sum_drop0 = sum_of_logs(attacking_scores[1:])
        print(f"attacking score log sum: {atk_log_sum}")
        # print(f"attacking score log sum drop value which index=0: {atk_log_sum_drop0}")

        # no attacking
        print(f"------------------------------- without adv prompt -------------------------------------")
        no_atk_string = f'{neg_dict["goal"][i]}'
        model_inputs, input_length = build_chat_input(no_atk_string, no_align_tokenizer)
        processor = ChangingSavingLogitsProcessor(aim=align_generated_ids[0])
        no_atk_generated_ids = no_align_model.generate(**model_inputs, logits_processor=[processor], max_new_tokens=200)
        # global true_no_atk_generated_ids
        # true_no_atk_generated_ids = no_atk_generated_ids[:, input_length:]
        # response = no_align_tokenizer.batch_decode(true_no_atk_generated_ids, skip_special_tokens=True)[0]
        # print(response)
        no_attacking_scores = processor.prob
        processor.prob = []

        print(f"no attacking score: {no_attacking_scores}")
        no_atk_log_sum = sum_of_logs(no_attacking_scores)
        # no_atk_log_sum_drop0 = sum_of_logs(no_attacking_scores[1:])
        print(f"no attacking score log sum: {no_atk_log_sum}")
        # print(f"no attacking score log sum drop value which index=0: {no_atk_log_sum_drop0}")
        negative_scores.append([atk_log_sum, no_atk_log_sum, atk_log_sum - no_atk_log_sum])

    print("positive sum(log(prob)) list, \n with adv prompt | without adv prompt | without - with\n", np.array(positive_scores))
    print("negative sum(log(prob)) list, \n with adv prompt | without adv prompt | without - with\n", np.array(negative_scores))


if __name__ == '__main__':
    main()

