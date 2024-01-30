import argparse
import sys
import math
import random
import json
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from baselines import compute_bm25_scores, compute_tfidf_scores, compute_sentence_bert_scores

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList)

from logitsProcessors import SavingLogitsProcessor, ChangingSavingLogitsProcessor

vicuna_list = [[0, 10, 17, 19, 20, 21, 29, 31, 33, 40, 43, 45, 48, 51, 53], [2, 5, 7, 9, 11, 12, 14, 16, 24, 26, 27, 34, 36, 38, 39]]
# vicuna_list = [[1, 10, 17, 19, 20, 21], [2, 5, 7, 9, 11, 12, 14]]
llama_list = [[13,32,34,35,36], [2,3,11,18,37], [5,6,7,8,9]]

_MODELS = {
    "llama": ["/data/bochuan/llm-attacks/DIR/llama/Llama-2-7b-chat-hf", "llama", llama_list],
    "vicuna": ["/data/bochuan/llm-attacks/DIR/vicuna/vicuna-7b-v1.3", "vicuna", vicuna_list],
    "guanaco": ["/data/bochuan/llm-attacks/DIR/guanaco-7B-HF", "guanaco"]
}

def sum_of_logs(lst):
    return math.exp(sum(math.log(x) for x in lst) / len(lst))

def rank_list(scores, words):
    
    ranked_score = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranked_words = [words[i] for i in ranked_score]
    return ranked_words

# Function to calculate mean, excluding None values
def mean(column):
    filtered = [x for x in column if x is not None]
    return sum(filtered) / len(filtered) if filtered else None

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

def masking_words(input, input_list, word_ids=0, method='drop', window_size=1, random_sample=False):

    if random_sample == False:
        words_masked = input_list[word_ids : (word_ids+window_size)]
        ids = list(range(word_ids, (word_ids+window_size)))
    else: 
        n = len(input_list)
        ids = random.sample(list(range(n)), window_size)
        words_masked = [input_list[i] for i in range(n) if i not in ids]

    if method == 'drop':
        # input_new = input.replace(words_masked, '')
        input_new = ' '.join(word for word in input_list if word not in words_masked)
    elif method == 'replace_#':
        for word in words_masked:
            input_new = input.replace(word, '#')
    else:
        input_new = input

    return input_new, words_masked, ids


def generate_input(args, input, processor, mask=False):
    model_inputs, input_length = tokenizer_query(input, tokenizer, args.test_model) # tokenize input strings
    generated_ids = model.generate(**model_inputs, logits_processor=[processor], max_new_tokens=args.max_new_tokens)
    true_generated_ids = generated_ids[:, input_length:] # delete the indices of the input strings
    response = tokenizer.batch_decode(true_generated_ids, skip_special_tokens=True)[0]
    probs = processor.prob[args.hand_cutting_num:]
    processor.prob = []
    print(f"probs: {probs}")
    log_sum = sum_of_logs(probs)
    print(f"generating score log sum: {log_sum}")

    if mask == True:
        mask_processor = SavingLogitsProcessor()
        mask_generated_ids = model.generate(**model_inputs, logits_processor=[mask_processor], max_new_tokens=args.max_new_tokens)
        mask_true_generated_ids = mask_generated_ids[:, input_length:]
        response = tokenizer.batch_decode(mask_true_generated_ids, skip_special_tokens=True)[0]
        mask_probs = mask_processor.prob[args.hand_cutting_num:]
        mask_processor.prob = []
        # print(f"prob of new generations: {mask_probs}")
    return response, true_generated_ids, probs, log_sum


def is_cal(fix_string, words_list, qa_dict, i, args, s=3, random_sample=False):

    print('------------------------------ no mask ------------------------------')
    model_inputs, input_length = tokenizer_query(fix_string, tokenizer, args.test_model)
    processor = SavingLogitsProcessor()
    response, true_fix_generated_ids, fix_scores, fix_log_sum = generate_input(args, fix_string, processor)

    print('------------------------------ mask ------------------------------')

    n = len(words_list)
    rs_list = []

    if random_sample == True:
        niters = 20
    else:
        niters = n-s+1
    
    for j in range(niters):
        print(j)

        mask_fix_log_sum_list = [None for _ in range(n)]
    
        new_string, mask_words, ids = masking_words(qa_dict['goal'][i], words_list, word_ids=j, window_size=s, random_sample = random_sample)
        print(mask_words)
        mask_fix_string = f'{new_string} {qa_dict["controls"][i]}'
        processor = ChangingSavingLogitsProcessor(aim=true_fix_generated_ids[0])
        mask_response, true_mask_fix_generated_ids, mask_fix_scores, mask_fix_log_sum = generate_input(args, mask_fix_string, processor, mask=True)
        
        if random_sample == False:
            mask_fix_log_sum_list[j:(j+s)] = [mask_fix_log_sum] * s
        else:
            for id in ids:
                mask_fix_log_sum_list[id] = mask_fix_log_sum
        rs_list.append(mask_fix_log_sum_list)

    rs_array = list(zip(*rs_list))
    mask_fix_log_sum_mean = [mean(column) for column in rs_array]
    
    scores_dif_list = [fix_log_sum - x for x in mask_fix_log_sum_mean]
    print(scores_dif_list)
    scores_div_list = [x/fix_log_sum for x in mask_fix_log_sum_mean]
    print(scores_div_list)
    scores_dif_list_abs = [abs(fix_log_sum - x) for x in mask_fix_log_sum_mean]
    print(scores_dif_list_abs)

    ranked_words_dif = rank_list(scores_dif_list, words_list)
    ranked_words_div = rank_list(scores_div_list, words_list)
    ranked_words_dif_abs = rank_list(scores_dif_list_abs, words_list)
    
    return ranked_words_dif, ranked_words_div, ranked_words_dif_abs, scores_dif_list, scores_div_list
    
def plot_s(data, words_list, figure_path, score_given_s):

    ylim = max(max(sublist) for sublist in score_given_s)

    # Check the number of columns
    num_columns = len(data[0])

    # Plotting
    if num_columns > 1:
        # If more than 10 columns, split into two subplots
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
        plt.setp(ax, xticks=range(1, len(data[0]) + 1), ylim=[0, ylim])

        # Divide data for two subplots
        mid_point = num_columns // 2
        data1 = data[:mid_point]
        data2 = data[mid_point:]
        legend1 = words_list[:mid_point]
        legend2 = words_list[mid_point:]

        # Plot first half
        for row in data1:
            ax[0].plot(range(1, len(row) + 1), row)

        ax[0].set_title('First Half of words')
        ax[0].set_xlabel('Number of Masked Words')
        ax[0].set_ylabel('Importance Score')
        ax[0].legend(legend1)

        # Plot second half
        for row in data2:
            ax[1].plot(range(1, len(row) + 1), row)

        ax[1].set_title('Second Half of words')
        ax[1].set_xlabel('Number of Masked Words')
        ax[1].set_ylabel('Importance Score')
        ax[1].legend(legend2)

    else:
        # If 10 or fewer columns, use a single plot
        plt.figure(figsize=(8, 6))
        for row in data:
            plt.plot(range(1, len(row) + 1), row)
        plt.title('Line Plot of Each Row')
        plt.xlabel('Number of Masked Words')
        plt.ylabel('Importance Score')
        plt.legend(words_list)
        plt.xticks(range(1, len(data[0]) + 1)) 

    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close() 
    plt.show()

parser = argparse.ArgumentParser(description='llm metrics')
parser.add_argument('--hand_cutting_num', type=int, default=1)
parser.add_argument('--max_new_tokens', type=int, default=100)
parser.add_argument('--test_model', type=str, default='llama')
parser.add_argument('--manual_seed', default=0, type=int)
parser.add_argument('--device', default='2', type=str)
args = parser.parse_args([])

os.environ["CUDA_VISIBLE_DEVICES"] = args.device

# loading model
model_dir = _MODELS[args.test_model][0]
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
tokenizer.padding_side = 'left'

qa_dict = {'goal': ['How to make a cake',
                    'Enhancing and Robustness in Deep Learning through Distributionally Robust Optimization and Regularization Techniques',
                    'Despite their immense size, whales demonstrate remarkable intelligence, including complex communication skills and social structures',
                    'On a sunny day, the happy family enjoyed a delightful picnic in the park, basking in the warmth and cheerfulness of the weather.',
                    'Celebrating Joy The Essence of Happiness, Birthdays, and Friends in Creating Memorable Moments'],
           'controls': ['Please give some instructions',
                        'please give an abstract based on the title in 50 words',
                        'Please paraphrase it',
                        'Please paraphrase it',
                        'In the bustling heart of the city, Emma planned a surprise birthday gathering for her best friend, Lily, embodying the essence of happiness, birthdays, and friendship. The room burst into laughter and cheers as Lily walked in, her face lighting up with joy, surrounded by close friends who filled the space with warmth and love. That evening, as they shared stories and danced under the twinkling lights, they created memories that would be cherished for years, a testament to the enduring power of friendship and the simple joy of celebrating life together.']}

for i in range(len(qa_dict['goal'])):
    is_given_s = []; is_given_s_div = []; score_given_s = []; score_given_s_div = []
    test_model = args.test_model
    fix_string = f'{qa_dict["goal"][i]} {qa_dict["controls"][i]}'
    words_list = qa_dict['goal'][i].split()
    for s in range(1, len(words_list)):
        qa_is_dict = {'original':[], 'ranked_dif':[], 'ranked_div':[], 'ranked_dif_abs':[]}

        ranked_words_dif, ranked_words_div, ranked_words_dif_abs, scores_dif_list, scores_div_list = is_cal(fix_string, words_list, qa_dict, i, args, s=s)
        qa_is_dict['original'].append(qa_dict['goal'][i])
        qa_is_dict['ranked_dif'].append(ranked_words_dif)
        qa_is_dict['ranked_div'].append(ranked_words_div)
        qa_is_dict['ranked_dif_abs'].append(ranked_words_dif_abs)

        is_given_s.append(qa_is_dict['ranked_dif'][0])
        is_given_s_div.append(qa_is_dict['ranked_div'][0])
        score_given_s.append(scores_dif_list)
        score_given_s_div.append(scores_div_list)

    # Transpose the data so that each row (now column) can be plotted as a separate line
    transposed_data = list(zip(*score_given_s))

    # # transposed_data = score_given_s
    figure_path = 'results/figures/case'+str(i)+'.png'
    plot_s(transposed_data, words_list, figure_path, score_given_s)
    
    