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
import re

def load_model(model_dir, model_dtype=torch.float16):
    config_kwargs = {"output_hidden_states": True}
    config = AutoConfig.from_pretrained(model_dir, **config_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", do_sample=False, torch_dtype=model_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def sentence_diff(cans, refs, diff_dict, type):
    if type == 'bert':
        # Calculate BERTScore
        P, R, F1 = score([cans], [refs], lang="en", verbose=True)
        diff_dict['bert_p'].append(P.item()); diff_dict['bert_r'].append(R.item()); diff_dict['bert_f'].append(F1.item())
    elif type == 'rouge':
        rouge = Rouge()
        # Calculate ROUGE scores
        scores = rouge.get_scores([cans], [refs])[0]
        diff_dict['rouge1_p'].append(scores['rouge-1']['p']); diff_dict['rouge1_r'].append(scores['rouge-1']['r']); diff_dict['rouge1_f'].append(scores['rouge-1']['f'])
        diff_dict['rouge2_p'].append(scores['rouge-2']['p']); diff_dict['rouge2_r'].append(scores['rouge-2']['r']); diff_dict['rouge2_f'].append(scores['rouge-2']['f'])
        diff_dict['rougel_p'].append(scores['rouge-l']['p']); diff_dict['rougel_r'].append(scores['rouge-l']['r']); diff_dict['rougel_f'].append(scores['rouge-l']['f'])
    elif type == 'bleu':
        bleu_score = sentence_bleu([refs], cans)
        diff_dict['bleu'].append(bleu_score)
    elif type == 'tfidf':
        # Create a TfidfVectorizer
        vectorizer = TfidfVectorizer()
        # Transform the sentences into TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform([refs, cans])
        # Compute the cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        diff_dict['tfidf'].append(similarity[0][0])
    elif type == 'sentenceBert':
        similarity = compute_sentence_bert_scores([cans], [refs])[0]
        diff_dict['sentenceBert'].append(similarity)
    
    return diff_dict

def sentence_diff_batch(cans, refs, diff_dict, type):
    if type == 'bert':
        # Calculate BERTScore
        P, R, F1 = score(cans, [refs]*len(cans), lang="en", verbose=True)
        diff_dict['bert_p']=P; diff_dict['bert_r']=R; diff_dict['bert_f']=F1
    elif type == 'rouge':
        rouge = Rouge()
        # Calculate ROUGE scores
        # scores_all = rouge.get_scores(cans, [refs]*len(cans))
        # for scores in scores_all:
        for can in cans:
            scores = rouge.get_scores([can], [refs])[0]
            diff_dict['rouge1_p'].append(scores['rouge-1']['p']); diff_dict['rouge1_r'].append(scores['rouge-1']['r']); diff_dict['rouge1_f'].append(scores['rouge-1']['f'])
            diff_dict['rouge2_p'].append(scores['rouge-2']['p']); diff_dict['rouge2_r'].append(scores['rouge-2']['r']); diff_dict['rouge2_f'].append(scores['rouge-2']['f'])
            diff_dict['rougel_p'].append(scores['rouge-l']['p']); diff_dict['rougel_r'].append(scores['rouge-l']['r']); diff_dict['rougel_f'].append(scores['rouge-l']['f'])
    elif type == 'bleu':
        for can in cans:
            bleu_score = sentence_bleu([refs], can)
            diff_dict['bleu'].append(bleu_score)
    elif type == 'tfidf':
        # Create a TfidfVectorizer
        # vectorizer = TfidfVectorizer(token_pattern=r'.')
        for can in cans:
            if len(can.split()) == 1:
                vectorizer = TfidfVectorizer(token_pattern=r'.')
            else:
                vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([refs, can])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            diff_dict['tfidf'].append(similarity[0][0])
    elif type == 'sentenceBert':
        similarity = compute_sentence_bert_scores(cans, [refs])
        diff_dict['sentenceBert']=similarity
    
    return diff_dict

def response_sim(response, mask_response, diff_dict, data_type='single'):

    '''
    Calculate the sentence similarity between the original response and the response of masked words
    '''
    if data_type == 'single':
        diff_dict = sentence_diff(mask_response, response, diff_dict, 'sentenceBert')
        diff_dict = sentence_diff(mask_response, response, diff_dict, 'rouge')
        diff_dict = sentence_diff(mask_response, response, diff_dict, 'bleu')
        diff_dict = sentence_diff(mask_response, response, diff_dict, 'tfidf')
    
    elif data_type == 'batch':
        diff_dict = sentence_diff_batch(mask_response, response, diff_dict, 'sentenceBert')
        diff_dict = sentence_diff_batch(mask_response, response, diff_dict, 'rouge')
        diff_dict = sentence_diff_batch(mask_response, response, diff_dict, 'bleu')
        diff_dict = sentence_diff_batch(mask_response, response, diff_dict, 'tfidf')

    
    return diff_dict

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


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True