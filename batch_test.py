import argparse
import sys
import math
import random
import json
import numpy as np
import pandas as pd
import os
from IPython.core.display import display, HTML
import torch
from baselines import compute_bm25_scores, compute_tfidf_scores, compute_sentence_bert_scores, kl_divergence

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList,
                          AutoConfig)

from logitsProcessors import (SavingLogitsProcessor, ChangingSavingLogitsProcessor,SavingLogitsProcessorAll, ChangingSavingLogitsProcessorBatch, 
                              ChangingSavingLogitsProcessorAll, LogitsProcessor, SavingLogitsProcessorBatch)
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from bert_score import score
from rouge import Rouge 
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity
import random
import copy
from IS_cal_functions import *
import re
from baseline_captum import captum_cal
from captum.attr import (FeatureAblation, LLMAttribution, TextTokenInput, TextTemplateInput)
import torch.optim as optim
from baseline_padding_clas import BaselinePadding


_MODELS = {
    "llama": ["/data/bochuan/llm-attacks/DIR/llama/Llama-2-7b-chat-hf", "llama"],
    "vicuna": ["/data/bochuan/llm-attacks/DIR/vicuna/vicuna-7b-v1.3", "vicuna"],
    "guanaco": ["/data/bochuan/llm-attacks/DIR/guanaco-7B-HF", "guanaco"]
}

parser = argparse.ArgumentParser(description='llm metrics')
parser.add_argument('--hand_cutting_num', type=int, default=1)
parser.add_argument('--max_new_tokens', type=int, default=150)
parser.add_argument('--test_model', type=str, default='llama')
parser.add_argument('--manual_seed', default=3, type=int)
parser.add_argument('--device', default='1,2', type=str)
parser.add_argument('--temperature', type=int, default=1)
parser.add_argument('--seed', type=int, default=123)
args = parser.parse_args([])
args, _ = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
# torch.cuda.set_device(int(args.device))
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# loading model
model_dir = _MODELS[args.test_model][0]
config_kwargs = {"output_hidden_states": True}
config = AutoConfig.from_pretrained(model_dir, **config_kwargs)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", do_sample=False)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
tokenizer.padding_side = 'left'
# tokenizer.add_special_tokens({'mask_token':'[MASK]'})
# model.resize_token_embeddings(len(tokenizer))
# model.config.mask_token_id = tokenizer.mask_token_id
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id


args.dir = "results"
args.filename = 'alpaca'
args.task = 'padding_paradigm'
args.gen_method = 'not fix_response'
args.num = 5
num_start_token = 23; num_end_token = 4

from datasets import load_dataset
dataset = load_dataset("yahma/alpaca-cleaned")
data = dataset['train']['instruction'][:args.num]

def run_experiment(data, model, tokenizer, args):
    padding_alg = BaselinePadding(data, model, tokenizer, args)
    padding_alg.cal()

n1 = 15

fix_control = 'Please answer the question in a 100 words\n\n'
def tokenizer_query(query, tokenizer, test_model, fix_control=fix_control):
    if test_model == "llama":
        input_string = f'''\n<</SYS>>\n\n{fix_control}{query}[/INST]'''
        model_inputs = tokenizer([input_string], truncation=False, return_tensors='pt').to("cuda")
    else:
        input_string = f'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.\n### USER: {query}\n### ASSISTANT:'
        model_inputs = tokenizer([input_string], padding=True,truncation=False, return_tensors='pt').to("cuda")

    # print(model_inputs['input_ids'].shape)
    input_length = model_inputs['input_ids'].shape[-1]
    return model_inputs, input_length

diff_dict = {'loglikelihood':[], 'tfidf':[], 'bleu':[], 
            #  'bert_p':[], 'bert_r':[], 'bert_f':[], 
            'sentenceBert':[],
            'rouge1_p':[], 'rouge1_r':[], 'rouge1_f':[],
            'rouge2_p':[], 'rouge2_r':[], 'rouge2_f':[],
            'rougel_p':[], 'rougel_r':[], 'rougel_f':[],
            'kl':[]}


def model_generate_batch(batch_input_ids, model, tokenizer, batch_processor, args):

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
    batch_response = tokenizer.batch_decode(_batch_true_generated_ids)
    
    return batch_probs, batch_logsum, batch_response


i = 4
words_list = data[i].split()
# if len(words_list)<n1 or len(words_list)>40:
#     continue

input = ' '.join(word for word in words_list)
fix_string = f'{input}'
print(f"### Input ### {fix_string}")

fix_model_inputs, fix_input_length = tokenizer_query(fix_string, tokenizer, args.test_model)
processor = SavingLogitsProcessor()
generated_ids = model.generate(**fix_model_inputs, logits_processor=[processor], max_new_tokens=args.max_new_tokens, 
                            num_beams=1, do_sample=False)
fix_true_generated_ids = generated_ids[:, fix_input_length:] # delete the indices of the input strings
fix_response = tokenizer.batch_decode(fix_true_generated_ids, skip_special_tokens=True)[0]
fix_probs = processor.prob[args.hand_cutting_num:]
processor.prob = []
# print(f"probs: {probs}")
fix_logsum = sum_of_logs(fix_probs)
if args.gen_method == 'fix_response':
    fix_response_ids = fix_true_generated_ids[0][:10]
    fix_response = tokenizer.batch_decode([fix_true_generated_ids[0][10:]], skip_special_tokens=True)[0]
else:
    fix_response = tokenizer.batch_decode([fix_true_generated_ids[0]], skip_special_tokens=True)[0]
print(f"### fix response ### {fix_response}")


# batch
all_ids = fix_model_inputs['input_ids']
start_ids = all_ids.squeeze(0)[:num_start_token]; end_ids = all_ids.squeeze(0)[-num_end_token:]
input_ids = all_ids.squeeze(0)[num_start_token:-num_end_token]
num_input_ids = len(input_ids)
window_size = 3
rs_list = []
niters = num_input_ids+window_size-1
niters = 4

batch_inputs = []
batch_processor = []
batch_size = niters
for j in range(batch_size):
    input_ids_masked, ids_masked = padding_input(input_ids.detach().cpu().tolist(), input_ids=j, window_size=window_size)
    input_ids_kept = torch.tensor([token for token in input_ids.detach().cpu().tolist() if token not in input_ids_masked]).to('cuda')
    new_input_ids = torch.cat([start_ids,input_ids_kept, end_ids]).unsqueeze(0)
    new_input = tokenizer.batch_decode(new_input_ids)[0]
    batch_inputs.append(new_input)
    # batch_processor.append(processor)

batch_input_ids = tokenizer(batch_inputs, padding=True, return_tensors="pt").to("cuda")
batch_processor = SavingLogitsProcessorBatch(batch_size=batch_size)
batch_probs, batch_logsum, batch_response = model_generate_batch(batch_input_ids, model, tokenizer, batch_processor, args)
for j in range(niters):
    new_logsum_list = [None for _ in range(num_input_ids)]
    input_ids_masked, ids_masked = padding_input(input_ids.detach().cpu().tolist(), input_ids=j, window_size=window_size)
    id1 = max(0, ids_masked[0]); id2 = min(ids_masked[-1], num_input_ids-1)
    new_logsum_list[id1 : (id2+1)] = [batch_logsum[j]] * (id2 - id1 + 1)
    rs_list.append(new_logsum_list)
rs_array = list(zip(*rs_list))
new_logsum_mean = [mean(column) for column in rs_array]

# scores_dif_list = [fix_logsum - x for x in new_logsum_mean]
# rank_scores_dif, rank_indices = torch.tensor(scores_dif_list).topk(num_input_ids)
# rank_tokens = [input_ids[idx].item() for idx in rank_indices]
# print(f'### Rank of Tokens ### {tokenizer.batch_decode(torch.tensor(rank_tokens))}')
print(new_logsum_mean[:4])


# individual
for j in range(niters):
    new_logsum_list = [None for _ in range(num_input_ids)]

    input_ids_masked, ids_masked = padding_input(input_ids.detach().cpu().tolist(), input_ids=j, window_size=window_size)
    input_ids_kept = torch.tensor([token for token in input_ids.detach().cpu().tolist() if token not in input_ids_masked]).to('cuda')
    new_input_ids = torch.cat([start_ids,input_ids_kept, end_ids]).unsqueeze(0)
    processor = SavingLogitsProcessor()
    # new_probs, new_logsum, new_response = model_generate(new_input_ids, model, tokenizer, processor, args)
    generated_ids = model.generate(batch_input_ids.input_ids[j].unsqueeze(0), logits_processor=[processor], max_new_tokens=args.max_new_tokens, num_beams=1, do_sample=False, 
                                   attention_mask=batch_input_ids.attention_mask[j].unsqueeze(0))
    true_generated_ids = generated_ids[:, batch_input_ids.input_ids[j].shape[0]:]
    new_probs = processor.prob[args.hand_cutting_num:]
    processor.prob = []
    new_logsum = sum_of_logs(new_probs)
    new_response = tokenizer.batch_decode(true_generated_ids, skip_special_tokens=True)[0]
    id1 = max(0, ids_masked[0]); id2 = min(ids_masked[-1], num_input_ids-1)
    new_logsum_list[id1 : (id2+1)] = [new_logsum] * (id2 - id1 + 1)
    rs_list.append(new_logsum_list)
rs_array = list(zip(*rs_list))
new_logsum_mean = [mean(column) for column in rs_array]
print(new_logsum_mean[:4])