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
from baseline.baselines import compute_bm25_scores, compute_tfidf_scores, compute_sentence_bert_scores, kl_divergence

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList,
                          AutoConfig)

from logitsProcessors import SavingLogitsProcessor, ChangingSavingLogitsProcessor, ChangingSavingLogitsProcessorBatch, SavingLogitsProcessorBatch
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
from baseline.baseline_captum import captum_cal
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
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True
set_seed(args.seed)

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


# In[3]:

args.dir = "results"
# args.filename = 'alpaca'
args.filename = 'boolq'
args.gen_method = 'not fix_response'
args.optim_method = 'pgd'
args.select_method = 'sample'
args.task = f'optim_{args.optim_method}_{args.select_method}_batch_2'
args.num = 50
args.n1 = 15
args.method = 'drop'
# num_start_token = 22; num_end_token = 4
num_token = {'start_token':{'Please answer this question in 100 words\n\n':22,
                             'Please answer the question in three sentences\n\n': 19,
                             "Please answer the question in three sentences, and start the response with: Sure, here is the anwser to your question:\n\n": 38,
                             "Please answer the question in three sentences, and start each sentence with a bullet point\n\n": 28},
            'end_token': {'Sure, here is the answer:\n\n': 14, 
                             '': 4}
            }

# from datasets import load_dataset
# dataset = load_dataset("yahma/alpaca-cleaned")
# num = len(dataset['train']['instruction'])
# data = dataset['train']['instruction'][:args.num]
data = ['Can I boil water to 10°C in regular at normal atmospheric pressure? Yes/No','Can you drink alcohol in public in denmark? Yes/No']


# fix_control1 = 'Please answer the question in three sentences\n\n'
fix_control1 = 'Please answer this question in 100 words\n\n'
# fix_control2 = 'Sure, here is the anwser of this question:'
# fix_control1 = "Please answer the question in three sentences, and start the response with: Sure, here is the anwser to your question:\n\n"
# fix_control1 = "Please answer the question in three sentences, and start each sentence with a bullet point\n\n"
fix_control2 = ''
# fix_control2 = 'Sure, here is the answer:\n\n'
def tokenizer_query(query, tokenizer, test_model, fix_control1=fix_control1, fix_control2=fix_control2):
    if test_model == "llama":
        input_string = f'''\n<</SYS>>\n\n{fix_control1}{query}[/INST]{fix_control2}'''
        model_inputs = tokenizer([input_string], truncation=False, return_tensors='pt').to("cuda")
    else:
        input_string = f'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.\n### USER: {query}\n### ASSISTANT:'
        model_inputs = tokenizer([input_string], padding=True,truncation=False, return_tensors='pt').to("cuda")

    # print(model_inputs['input_ids'].shape)
    input_length = model_inputs['input_ids'].shape[-1]
    return model_inputs, input_length

num_start_token = num_token['start_token'][fix_control1]
num_end_token = num_token['end_token'][fix_control2]
num_end_token = num_end_token + 3

# optim_iters = 150

# In[29]:
# fix_control = 'Please answer the question in a 100 words\n\n'
# def tokenizer_query(query, tokenizer, test_model, fix_control=fix_control):
#     if test_model == "llama":
#         input_string = f'''\n<</SYS>>\n\n{fix_control}{query}[/INST]'''
#         model_inputs = tokenizer([input_string], truncation=False, return_tensors='pt').to("cuda")
#     else:
#         input_string = f'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.\n### USER: {query}\n### ASSISTANT:'
#         model_inputs = tokenizer([input_string], padding=True,truncation=False, return_tensors='pt').to("cuda")

#     # print(model_inputs['input_ids'].shape)
#     input_length = model_inputs['input_ids'].shape[-1]
#     return model_inputs, input_length

def model_optim(model, mask_input):
    # model.layers
    mask_input_temp = mask_input
    for i, layers in enumerate(model.model.layers):
        mask_output_temp = model.model.layers[i](mask_input_temp)
        # import ipdb; ipdb.set_trace()
        mask_input_temp = mask_output_temp[0]
    # model.norm
    mask_output_temp = model.model.norm(mask_input_temp)
    # model.lm_head
    mask_output = model.lm_head(mask_output_temp)

    # caculate the logits
    # import ipdb; ipdb.set_trace()
    mask_output = mask_output.log_softmax(-1).squeeze(0).to('cuda')
    mask_selected_probs = mask_output[:-1][torch.arange(len(selected_ids)), selected_ids]
    mask_logits = mask_selected_probs[(fix_input_length-1):]

    return mask_logits

# def model_generate(input_ids, model, tokenizer, processor, args):
#     generated_ids = model.generate(input_ids, logits_processor=[processor], max_new_tokens=args.max_new_tokens, num_beams=1, do_sample=False)
#     true_generated_ids = generated_ids[:, input_ids.shape[1]:]
#     probs = processor.prob[args.hand_cutting_num:]
#     processor.prob = []
#     logsum = sum_of_logs(probs)
#     response = tokenizer.batch_decode(true_generated_ids, skip_special_tokens=True)[0]

#     return probs, logsum, response

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
    batch_response = tokenizer.batch_decode(_batch_true_generated_ids, skip_special_tokens=True)
    
    return batch_probs, batch_logsum, batch_response


# def bicut(m, k, ub=1):
#     m_cut = torch.clip(m, min=0, max=ub)
#     if torch.sum(m) >= k:
#         m_update = m_cut
#     else:
#         mu_l = m.min().item() - 1
#         mu_u = m.max().item()
#         mu_hat = (mu_l + mu_u) / 2
#         m_update = torch.clip(m - mu_hat, min=0, max=ub)

#     return m_update

def bicut(m, k, xi, ub=1):

    m_cut = torch.clip(m, min=0, max=ub)
    
    if torch.sum(m_cut) >= k:
        m_update = m_cut
    else:
        mu_l = m.min().item()
        mu_u = m.max().item()
        while np.abs(mu_u - mu_l) > xi:
            mu_mid = (mu_l + mu_u) / 2
            g_mu_mid = torch.sum(torch.clip(m+mu_mid, 0, ub)) - k
            g_mu_l = torch.sum(torch.clip(m+mu_l, 0, ub)) - k
            if g_mu_mid == 0:
                break
            if torch.sign(g_mu_mid) == torch.sign(g_mu_l):
                mu_l = mu_mid
            else:
                mu_u = mu_mid
                    
        m_update = torch.clip(m + mu_mid, min=0, max=ub)

    return m_update


dir = "results"
diff_dict = {'loglikelihood':[], 'tfidf':[], 'bleu':[], 
            'sentenceBert':[],
            'rouge1_p':[], 'rouge1_r':[], 'rouge1_f':[],
            'rouge2_p':[], 'rouge2_r':[], 'rouge2_f':[],
            'rougel_p':[], 'rougel_r':[], 'rougel_f':[],
            'kl':[]}

# Creating deep copies for each dictionary
random_drop_diff_dict = copy.deepcopy(diff_dict)
drop_diff_dict = copy.deepcopy(diff_dict)
# rev_drop_diff_dict = copy.deepcopy(diff_dict)
# captum_drop_diff_dict = copy.deepcopy(diff_dict)

diff_dict_name = {'tfidf':['TF-IDF'], 'bleu':['BLEU'], 
                'sentenceBert':['Sentence Bert'],
                'rouge1_p':['Rouge-1 Precision'], 'rouge1_r':['Rouge-1 Recall'], 'rouge1_f':['Rouge-1 F1_score'],
                'rouge2_p':['Rouge-2 Precision'], 'rouge2_r':['Rouge-2 Recall'], 'rouge2_f':['Rouge-2 F1_score'],
                'rougel_p':['Rouge-L Precision'], 'rougel_r':['Rouge-L Recall'], 'rougel_f':['Rouge-L F1_score'], 
                'loglikelihood':['Log Likelihood'], 'kl':['KL-Divergence']}

diff_dict_keys = diff_dict.keys()

loss_fig_path = f'{args.dir}/{args.task}'
if not os.path.exists(loss_fig_path):
    os.makedirs(loss_fig_path)

# In[80]:

for j in range(len(data)):
    words_list = data[j].split()
    # if len(words_list)<args.n1 or len(words_list)>40:
    #     continue
    input_sentence = ' '.join(word for word in words_list)
    fix_string = f'{input_sentence}'
    print(f"### Input ### {fix_string}")

    # Generate the original response
    fix_model_inputs, fix_input_length = tokenizer_query(fix_string, tokenizer, args.test_model)
    processor = SavingLogitsProcessor()
    fix_generated_ids = model.generate(**fix_model_inputs, logits_processor=[processor], max_new_tokens=args.max_new_tokens, 
                                    num_beams=1, do_sample=False)
    fix_true_generated_ids = fix_generated_ids[:, fix_input_length:] # delete the indices of the input strings
    if args.gen_method == 'fix_response':
        fix_response_ids = fix_true_generated_ids[0][:10]
        fix_response = tokenizer.batch_decode([fix_true_generated_ids[0][10:]], skip_special_tokens=True)[0]
    else:
        fix_response = tokenizer.batch_decode([fix_true_generated_ids[0]], skip_special_tokens=True)[0]
    fix_probs = processor.prob[args.hand_cutting_num:]
    processor.prob = []
    # print(f"probs: {probs}")
    fix_log_sum = sum_of_logs(fix_probs)

    # # Append the original response and re-generate
    # new_string = fix_string + fix_response
    # model_inputs, input_length = tokenizer_query(new_string, tokenizer, args.test_model)

    # # num_start_token = 10; num_end_token = 4
    # all_ids = model_inputs['input_ids']
    # start_ids = all_ids.squeeze(0)[:num_start_token]; end_ids = all_ids.squeeze(0)[-num_end_token:]
    # input_ids = all_ids.squeeze(0)[num_start_token:(fix_input_length-num_end_token)] # delete the indices at the beginning
    # output_ids = all_ids.squeeze(0)[(fix_input_length-num_end_token):-num_end_token] # delete the indices in the end
    all_ids = fix_generated_ids
    start_ids = all_ids.squeeze(0)[:num_start_token]
    input_ids = all_ids.squeeze(0)[num_start_token:(fix_input_length - num_end_token)]
    end_ids = all_ids.squeeze(0)[(fix_input_length-num_end_token):fix_input_length]
    output_ids = all_ids.squeeze(0)[fix_input_length:]

    # outputs = model(all_ids, return_dict=True)
    # outputs = outputs.logits.squeeze(0).log_softmax(-1)

    selected_ids = all_ids.squeeze(0)[1:].long() # all indices except the first one: (x2, x3,..., xn, y1, y2,..., ym, y(m+1),...,y(m+4))
    # selected_probs = outputs[:-1][torch.arange(len(selected_ids)), selected_ids]
    # logits = selected_probs[((len(input_ids))+num_start_token):-num_end_token].sum()

    token_groups = divide_into_groups(len(input_ids), m=10)  
    # import ipdb; ipdb.set_trace()
    # random_drop_diff_dict_temp = copy.deepcopy(diff_dict)
    drop_diff_dict_temp = copy.deepcopy(diff_dict)

    k = len(input_ids)
    fix_input_ids = fix_model_inputs['input_ids']
    input_ids_kept = fix_input_ids
    # probability(y|x_mask)
    # processor = SavingLogitsProcessor()
    # new_probs, new_logsum, new_response = model_generate(input_ids_kept, model, tokenizer, processor, args)
    # # probability(y|x)
    # temp_processor = ChangingSavingLogitsProcessor(aim=fix_true_generated_ids[0])
    # temp_probs, temp_logsum, _, = model_generate(input_ids_kept, model, tokenizer, temp_processor, args)

    # # drop/mask keywords
    # drop_diff_dict_temp = response_sim(fix_response, new_response, drop_diff_dict_temp)
    # drop_diff_dict_temp['loglikelihood'].append(temp_logsum)
    # # random_drop_diff_dict_temp = response_sim(fix_response, new_response, random_drop_diff_dict_temp)
    # # random_drop_diff_dict_temp['loglikelihood'].append(temp_logsum)
    # drop_diff_dict_temp['kl'].append(kl_divergence([x/sum(fix_probs) for x in fix_probs], [x/sum(temp_probs) for x in temp_probs]))
    # random_drop_diff_dict_temp['kl'].append(kl_divergence([x/sum(fix_probs) for x in fix_probs], [x/sum(temp_probs) for x in temp_probs]))


    batch_inputs_mask = []
    new_input_mask = tokenizer.batch_decode(input_ids_kept[0][1:].unsqueeze(0))[0]
    batch_inputs_mask.append(new_input_mask)
    for l in range(len(token_groups)-1):
        k = k - len(token_groups[l])
        if l == 0:
            lr = 0.5
            lr = 2e-5
            optim_iters = 1000
        else:
            lr = .01
            optim_iters = 200
        print(k)
        # ------------------------------ Mask Optimization ------------------------------
        # lr = .01
        lda = 0.25 # 0.2
        bta=0.1
        # mask = torch.sigmoid(torch.rand(len(input_ids))).to(dtype=torch.float32)
        mask = torch.rand(len(input_ids))
        mask.requires_grad = True
        # s = len(mask)-1
        optimizer = optim.Adam([mask], lr=lr)
        losses = []

        loss_final = 0
        for i in range(optim_iters):
            optimizer.zero_grad()

            # model.embed_tokens
            input = model.model.embed_tokens(all_ids).squeeze(0)
            mask_start = torch.ones(num_start_token)
            mask_control = torch.ones(num_end_token)
            mask_end = torch.ones(len(output_ids))
            # print('mask', mask)
            mask_matrix = torch.diag(torch.cat((mask_start, mask, mask_control, mask_end), dim=0)).to('cuda')
            # print('diag', mask_matrix.diag())
            mask_input = torch.matmul(mask_matrix, input).unsqueeze(0)
            # mask_input = input.unsqueeze(0)

            mask_logits = model_optim(model, mask_input)

            if args.optim_method == 'soft':
                loss = mask_logits-sum(fix_probs) + lda * (k-sum(mask)) + bta * (sum(mask)-s)
            elif args.optim_method == 'hard':
                loss = mask_logits
            elif args.optim_method == 'prob':
                loss = mask_logits-sum(fix_probs) + lda * (k-sum(mask)) 
            elif args.optim_method == 'pgd':
                loss = torch.sum(mask_logits)

            loss.backward()
            optimizer.step()

            # import ipdb; ipdb.set_trace()

            if args.optim_method == 'soft':
                with torch.no_grad():
                    mask.clamp_(0,1)
            elif args.optim_method == 'hard':
                with torch.no_grad():
                    mask_data, indices = torch.topk(mask.data, k)
                    new_mask_data = torch.zeros_like(mask)
                    top_values_greater_than_threshold = mask_data > 0.05
                    new_mask_data.scatter_(0, indices[top_values_greater_than_threshold], 1)
                    mask.data = new_mask_data  
            elif args.optim_method == 'prob':
                with torch.no_grad():
                    mask.clamp_(0,1)
            elif args.optim_method == 'pgd':
                with torch.no_grad():
                    # mask.data = bicut(mask.data, k, 1e-4)
                    # with torch.no_grad():
                    mask.clamp_(0,1)
            
            # print('mask', mask)
        
            losses.append(loss.item())

            if i % 20 == 0:
                print(f"Iteration {i}, Loss: {loss.item()}, Constraint: {sum(mask).item()}")
            if loss < loss_final and i > 0:
                mask_temp_final = mask.data

        # if k > len(input_ids) - len(token_groups[0])*2:
        mask.data = mask_temp_final

        # Plot and save loss curve
        plt.figure()
        plt.plot(losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.savefig(f'{loss_fig_path}/loss_curve_{j}.png')
        plt.close()
        
        if args.select_method == 'sample':
            # mask_update = mask.data[:]
            # mask_loss = 1e10
            # for _ in range(20):
            #     mask_sample = torch.bernoulli(mask.data)
            #     if sum(mask_sample) in [k-2, k+2]:
            #         processor = ChangingSavingLogitsProcessor(aim=fix_true_generated_ids[0])
            #         sample_input_ids = torch.cat([start_ids, input_ids[mask_sample==1], end_ids]).unsqueeze(0)
            #         sample_generated_ids = model.generate(sample_input_ids, logits_processor=[processor], max_new_tokens=args.max_new_tokens, 
            #                                         num_beams=1, do_sample=False)
            #         sample_true_generated_ids = sample_generated_ids[:, sample_input_ids.shape[1]:] # delete the indices of the input strings
            #         # sample_response = tokenizer.batch_decode(sample_true_generated_ids, skip_special_tokens=True)[0]
            #         sample_probs = processor.prob[args.hand_cutting_num:]
            #         processor.prob = []
            #         # print(f"probs: {probs}")
            #         sample_log_sum = sum_of_logs(sample_probs)
            #         if mask_loss > sample_log_sum:
            #             mask_loss = sample_log_sum
            #             mask_update = mask_sample
            # idmask = (mask_update == 1)
            probabilities = mask.data[:]
            n_sample = 20
            expanded_probabilities = probabilities.repeat(n_sample, 1).T
            samples = torch.bernoulli(expanded_probabilities)
            sample_input_batch = []
            mask_sample_batch = []
            for i in range(n_sample):
                mask_sample = samples[:, i]
                if mask_sample.sum().item() in [k-1, k+1]:
                    sample_input_ids = torch.cat([start_ids[1:], input_ids[mask_sample==1], end_ids]).unsqueeze(0)
                    sample_input = tokenizer.batch_decode(sample_input_ids)[0]
                    sample_input_batch.append(sample_input)
                    mask_sample_batch.append(mask_sample)

            if len(mask_sample_batch) > 0:
                sample_ids_batch = tokenizer(sample_input_batch, padding=True, return_tensors="pt").to("cuda")
                batch_processor_change = ChangingSavingLogitsProcessorBatch(aim=fix_true_generated_ids[0], batch_size=sample_ids_batch['input_ids'].shape[0])
                temp_batch_probs, temp_batch_logsum, temp_batch_response = model_generate_batch(sample_ids_batch, model, tokenizer, batch_processor_change, args)
                min_val, min_idx = torch.min(torch.tensor(temp_batch_logsum), 0)
                idmask = (mask_sample_batch[min_idx] == 1)
            else: 
                mask_update = mask.data[:]
                idmask = torch.zeros_like(mask_update)
                idmask[torch.topk(mask_update, k).indices] = 1
                idmask = (idmask == 1)

        elif args.select_method == 'topk':
            mask_update = torch.zeros_like(mask.data)
            mask_update[torch.topk(mask.data, k).indices] = 1
            idmask = (mask_update == 1)
        else: 
            idmask = (mask.data == 1)
        
        # ------------------------------ Drop Sequentially ------------------------------"
    
        if args.method == 'drop':
            input_ids_kept_temp = input_ids[idmask]
        elif args.method == 'mask':
            input_ids_kept_temp = torch.where(idmask.to('cuda'), input_ids, torch.tensor(tokenizer.mask_token_id, dtype=input_ids.dtype))
        input_ids_kept = torch.cat((start_ids[1:], input_ids_kept_temp, fix_input_ids.squeeze(0)[-num_end_token:]), dim=0).unsqueeze(0)
        if args.gen_method == 'fix_response':
            input_ids_kept = torch.cat((input_ids_kept[0], fix_response_ids), dim=0).unsqueeze(0)
        print(f"### Optimization ### {tokenizer.batch_decode([input_ids_kept_temp])[0]}")
        print(f"K = {idmask.sum()}")
        new_input_mask = tokenizer.batch_decode(input_ids_kept)[0]
        batch_inputs_mask.append(new_input_mask)

        # # probability(y|x_mask)
        # processor = SavingLogitsProcessor()
        # # new_probs, new_logsum, new_response = model_generate(input_ids_kept, model, tokenizer, processor, args)
        # if args.gen_method == 'fix_response':
        #     input_ids_kept_fix = torch.cat((input_ids_kept[0], fix_response_ids), dim=0).unsqueeze(0)
        #     new_probs, new_logsum, new_response = model_generate(input_ids_kept_fix, model, tokenizer, processor, args)
        # else:
        #     new_probs, new_logsum, new_response = model_generate(input_ids_kept, model, tokenizer, processor, args)
        # print(f'### Output ###: {new_response}')
        # # probability(y|x)
        # temp_processor = ChangingSavingLogitsProcessor(aim=fix_true_generated_ids[0])
        # temp_probs, temp_logsum, _, = model_generate(input_ids_kept, model, tokenizer, temp_processor, args)

        # # drop/mask keywords
        # drop_diff_dict_temp = response_sim(fix_response, new_response, drop_diff_dict_temp)
        # drop_diff_dict_temp['loglikelihood'].append(temp_logsum)
        # drop_diff_dict_temp['kl'].append(kl_divergence([x/sum(fix_probs) for x in fix_probs], [x/sum(temp_probs) for x in temp_probs]))
    batch_size = len(token_groups)
    batch_new_input_ids = tokenizer(batch_inputs_mask, padding=True, return_tensors="pt").to("cuda")
    batch_processor = SavingLogitsProcessorBatch(batch_size=batch_size)
    new_batch_probs, new_batch_logsum, new_batch_response = model_generate_batch(batch_new_input_ids, model, tokenizer, batch_processor, args)
    batch_processor_change = ChangingSavingLogitsProcessorBatch(aim=fix_true_generated_ids[0], batch_size=batch_size)
    temp_batch_probs, temp_batch_logsum, _ = model_generate_batch(batch_new_input_ids, model, tokenizer, batch_processor_change, args)

    drop_diff_dict_temp = response_sim(fix_response, new_batch_response, diff_dict=drop_diff_dict_temp, data_type='batch')
    drop_diff_dict_temp['loglikelihood'] = temp_batch_logsum
    for temp_probs in temp_batch_probs:
        drop_diff_dict_temp['kl'].append(kl_divergence([x/sum(fix_probs) for x in fix_probs], [x/sum(temp_probs) for x in temp_probs]))
    drop_diff_dict_temp['response'] = new_batch_response


    # ------------------------------ Drop Randomly ------------------------------
    
    # ids_indices = list(range(input_ids.size(0)))
    # random.shuffle(ids_indices)

    # indices = torch.tensor([]).to(dtype=int)
    # for k2 in range(len(token_groups)):
        
    #     idmask = torch.ones_like(input_ids, dtype=torch.bool)  # Start with all True
    #     for val in input_ids[indices]:
    #         idmask &= (input_ids != val)  # Set to False for values to remove

    #     if method == 'drop':
    #         input_ids_kept_temp = input_ids[idmask]
    #     elif method == 'mask':
    #         input_ids_kept_temp = torch.where(idmask.to('cuda'), input_ids, torch.tensor(tokenizer.mask_token_id, dtype=input_ids.dtype))
    #     input_ids_kept = torch.cat((start_ids, input_ids_kept_temp, fix_input_ids.squeeze(0)[-num_end_token:]), dim=0).unsqueeze(0)

    #     id_groups = token_groups[k2]
    #     indices = torch.cat((indices, torch.tensor([ids_indices[k] for k in id_groups])), dim=0).to(dtype=int) # select top k indices

    #     # probability(y|x_mask)
    #     processor = SavingLogitsProcessor()
    #     new_probs, new_logsum, new_response = model_generate(input_ids_kept, model, tokenizer, processor, args)
    #     # probability(y|x)
    #     temp_processor = ChangingSavingLogitsProcessor(aim=fix_true_generated_ids[0])
    #     temp_probs, temp_logsum, _, = model_generate(input_ids_kept, model, tokenizer, temp_processor, args)

    #     # drop/mask keywords
    #     random_drop_diff_dict_temp = response_sim(fix_response, new_response, random_drop_diff_dict_temp)
    #     random_drop_diff_dict_temp['loglikelihood'].append(temp_logsum)
    #     random_drop_diff_dict_temp['kl'].append(kl_divergence([x/sum(fix_probs) for x in fix_probs], [x/sum(temp_probs) for x in temp_probs]))


    for key in diff_dict_keys:
        drop_diff_dict[key].append(drop_diff_dict_temp[key])
        # add_diff_dict[key].append(add_diff_dict_temp[key])
        # random_add_diff_dict[key].append(random_add_diff_dict_temp[key])
        # random_drop_diff_dict[key].append(random_drop_diff_dict_temp[key])
        # rev_drop_diff_dict[key].append(rev_drop_diff_dict_temp[key])
        # captum_drop_diff_dict[key].append(captum_drop_diff_dict_temp[key])

    # diff_dict_all = {'index':i, 'random':random_drop_diff_dict, 'is': drop_diff_dict, 'is_rev': rev_drop_diff_dict, 'captum':captum_drop_diff_dict}
    # diff_dict_all = {'index':i, 'random':random_drop_diff_dict, 'is': drop_diff_dict}

# output_all = {'is_drop': drop_diff_dict, 'random_drop':random_drop_diff_dict, 'rev_is_drop':rev_drop_diff_dict, 'captum': captum_drop_diff_dict}
output_all = {'is_drop': drop_diff_dict}
with open(f"{args.dir}/output/score_{args.filename}_{args.task}_{args.num}.json", 'w') as file:  
    json.dump(output_all, file, indent=4)
# plot_random_subplot(output_all, diff_dict_name, task, filename, group, num, dir)