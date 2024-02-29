#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

from logitsProcessors import SavingLogitsProcessor, ChangingSavingLogitsProcessor
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from bert_score import score
from rouge import Rouge 
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity
import random
import copy
# from IS_cal_functions import *
import re
from basline_captum import captum_cal
from captum.attr import (FeatureAblation, LLMAttribution, TextTokenInput, TextTemplateInput)
import torch.optim as optim


# In[2]:


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
parser.add_argument('--device', default='0,1', type=str)
parser.add_argument('--temperature', type=int, default=1)
parser.add_argument('--seed', type=int, default=123)
args = parser.parse_args([])
# args, _ = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
# torch.cuda.set_device(int(args.device))
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# loading model
model_dir = _MODELS[args.test_model][0]
config_kwargs = {
        "output_hidden_states": True
    }
config = AutoConfig.from_pretrained(model_dir, **config_kwargs)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", do_sample=False)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
tokenizer.padding_side = 'left'


# In[3]:


from datasets import load_dataset
filename = 'alpaca'
num = 5
group = "all"
dataset = load_dataset("yahma/alpaca-cleaned")
num = 20
data = dataset['train']['instruction'][:num]
data = [data[i] for i in [2, 4, 5, 7]]


# In[4]:


def tokenizer_query(query, tokenizer, test_model):
    if test_model == "llama":
        input_string = f'''\n<</SYS>>\n\n{query}[/INST]'''
        model_inputs = tokenizer([input_string], truncation=False, return_tensors='pt').to("cuda")
    else:
        input_string = f'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.\n### USER: {query}\n### ASSISTANT:'
        model_inputs = tokenizer([input_string], padding=True,truncation=False, return_tensors='pt').to("cuda")

    # print(model_inputs['input_ids'].shape)
    input_length = model_inputs['input_ids'].shape[-1]
    return model_inputs, input_length

def sum_of_logs(lst):
    return math.exp(sum(math.log(x) for x in lst) / len(lst))

def mask_words(word_list, binary_mask):
    # masked_list = [word if mask == 0 else '*' * len(word) for word, mask in zip(word_list, binary_mask)]
    masked_list = [word if mask == 1 else ''* len(word) for word, mask in zip(word_list, binary_mask)]
    return masked_list


# In[80]:


words_list = data[0].split()
input_sentence = ' '.join(word for word in words_list)
fix_string = f'{input_sentence}'

fix_model_inputs, fix_input_length = tokenizer_query(fix_string, tokenizer, args.test_model)
processor = SavingLogitsProcessor()
generated_ids = model.generate(**fix_model_inputs, logits_processor=[processor], max_new_tokens=args.max_new_tokens, 
                                num_beams=1, do_sample=False)
fix_true_generated_ids = generated_ids[:, fix_input_length:] # delete the indices of the input strings
fix_response = tokenizer.batch_decode(fix_true_generated_ids, skip_special_tokens=True)[0]
probs = processor.prob[args.hand_cutting_num:]
processor.prob = []
# print(f"probs: {probs}")
log_sum = sum_of_logs(probs)

new_string = fix_string + fix_response
model_inputs, input_length = tokenizer_query(new_string, tokenizer, args.test_model)

num_start_token = 10; num_end_token = 4
all_ids = model_inputs['input_ids']
input_ids = all_ids.squeeze(0)[num_start_token:(fix_input_length-num_end_token+1)] # delete the indices at the beginning
output_ids = all_ids.squeeze(0)[(fix_input_length-num_end_token+1):-num_end_token] # delete the indices in the end


# In[6]:


outputs = model(all_ids, return_dict=True)
outputs = outputs.logits.squeeze(0).log_softmax(-1)

selected_ids = all_ids.squeeze(0)[1:].long() # all indices except the first one: (x2, x3,..., xn, y1, y2,..., ym, y(m+1),...,y(m+4))
selected_probs = outputs[:-1][torch.arange(len(selected_ids)), selected_ids]
logits = selected_probs[(fix_input_length-num_start_token):-num_end_token].sum()


# In[96]:


mask = torch.sigmoid(torch.rand(len(input_ids))).to(dtype=torch.float32)
mask.requires_grad = True
mask_start = torch.ones(num_start_token); mask_output = torch.ones(num_end_token); mask_end = torch.ones(len(output_ids))
mask_matrix = torch.diag(torch.cat((mask_start, mask, mask_output, mask_end), dim=0)).to('cuda')


# In[97]:


lr = .1
k = 6
lda = 0.5
optimizer = optim.Adam([mask], lr=lr)
losses = []


# In[98]:


for i in range(50):
    optimizer.zero_grad()

    # model.embed_tokens
    input = model.model.embed_tokens(all_ids).squeeze(0)
    mask_input = torch.matmul(mask_matrix, input).unsqueeze(0)
    mask_input = input.unsqueeze(0)
    # model.layers
    mask_input_temp = mask_input
    for i, layers in enumerate(model.model.layers):
        mask_output_temp = model.model.layers[i](mask_input_temp)
        mask_input_temp = mask_output_temp[0]
    # model.norm
    mask_output_temp = model.model.norm(mask_input_temp)
    # model.lm_head
    mask_output = model.lm_head(mask_output_temp)

    # caculate the logits
    mask_output = mask_output.log_softmax(-1).squeeze(0).to('cuda')
    mask_selected_probs = mask_output[:-1][torch.arange(len(selected_ids)), selected_ids]
    mask_logits = mask_selected_probs[(fix_input_length-num_start_token):-num_end_token].sum()

    loss = mask_logits-sum(probs) + lda * (k - sum(mask))

    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if i % 10 == 0:
        print(f"Iteration {i}, Loss: {loss.item()}, Constraint: {sum(mask).item()}")

# Plot and save loss curve
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.savefig('loss_curve.png')
plt.show()



# # model.embed_tokens
# input = model.model.embed_tokens(all_ids).squeeze(0)
# mask = torch.ones(all_ids.shape[1], require_grad=True)
# mask_matrix = torch.diag(mask).to('cuda')
# mask_input = torch.matmul(mask_matrix, input).unsqueeze(0)
# mask_input = input.unsqueeze(0)
# # model.layers
# mask_input_temp = mask_input
# for i, layers in enumerate(model.model.layers):
#     mask_output_temp = model.model.layers[i](mask_input_temp)
#     mask_input_temp = mask_output_temp[0]
# # model.norm
# mask_output_temp = model.model.norm(mask_input_temp)
# # model.lm_head
# mask_output = model.lm_head(mask_output_temp)

# mask_output = mask_output.log_softmax(-1).squeeze(0).to('cuda')
# mask_selected_probs = mask_output[:-1][torch.arange(len(selected_ids)), selected_ids]
# mask_logits = mask_selected_probs[(fix_input_length-num_start_token):-num_end_token].sum()

# In[ ]:




