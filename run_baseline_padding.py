import os
import argparse
import random
import numpy as np
import os
import torch
import random
from baseline_padding_random import BaselinePadding
from datasets import load_dataset
from utils.text_scores import *
from utils.baseline_utils import *
from utils.search_utils import *

def run_experiment(data, model, tokenizer, args, seed):

    if args.task == 'random':
        alg = BaselinePadding(data, model, tokenizer, args)
        alg.cal()
    elif args.task == 'integrated_gradient':
        alg = BaselineIntegratedGradient(data, model, tokenizer, args)
        alg.cal()
    elif args.task == 'search_token_alg':
        alg = Gradient_Sample(data, model, tokenizer, args)
        alg.cal()
   
    print(np.mean(alg.use_time_list))


def main(args):
    _MODELS = {
        "llama_7b": ["meta-llama/Llama-2-7b-chat-hf", "llama_7b"],
        "vicuna": ["lmsys/vicuna-13b-v1.3", "vicuna"],
        "llama_13b": ["meta-llama/Llama-2-13b-chat-hf", "llama_13b"],
        "llama_70b": ["meta-llama/Llama-2-70b-chat-hf", "llama_70b"]
    }

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    set_seed(args.seed)
    model_dir = _MODELS[args.test_model][0]
    model, tokenizer = load_model(model_dir)

    if args.filename == 'alpaca':
        dataset = load_dataset("yahma/alpaca-cleaned")
        args.num = 1500
        data_instruction_all = dataset['train']['instruction'][:args.num]
        string_length = torch.tensor([len(data.split()) for data in data_instruction_all])
        indices = torch.where((string_length>15) & (string_length<40))[0]
        data_instruction = [data_instruction_all[idx] for idx in indices]
        data_input = [dataset['train']['input'][:args.num][idx] for idx in indices]
        data = [data_instruction[i]+ ' ' + data_input[i] for i in range(len(data_instruction))]
        args.fix_control1 = 'You are a helpful, respectful and honest assistant. Please analyze the following context and answer the question. '
        args.fix_control2 = ''
        args.fix_control3 = "Of course! I'd be happy to help you with that."
        data = data[:args.num]

    elif args.filename == 'tldr':
        random.seed(123)
        args.num = 120
        with open(f'/data/tldr_news_{args.num}_123.txt', 'r') as file:
            data = [line.strip() for line in file]
        args.fix_control1 = 'Please generate a headline from the following piece of context. '
        args.fix_control2 = ''
        args.fix_control3 = "Of course! I'd be happy to help you with that."

    elif args.filename == 'med':
        args.num = 100
        with open(f'/data/mental_health_{args.num}_123.txt', 'r') as file:
            data = [line.strip() for line in file]
        args.fix_control1 = 'Please generate advice or suggestions in response to the following mental health-related question. '
        args.fix_control2 = ''
        args.fix_control3 = "Of course! I'd be happy to help you with that."

    seed = args.seed
    run_experiment(data, model, tokenizer, args, seed)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='llm explain')
    parser.add_argument('--hand_cutting_num', required=False, type=int, default=1)
    parser.add_argument('--max_new_tokens', required=False, type=int, default=100)
    parser.add_argument('--test_model', required=False, type=str, default='llama_7b')
    parser.add_argument('--manual_seed', required=False, default=3, type=int)
    parser.add_argument('--device', required=False, default='3', type=str)
    parser.add_argument('--temperature', required=False, type=int, default=1)
    parser.add_argument('--seed', required=False, type=int, default=123)
    parser.add_argument('--num', required=False, type=int, default=1)
    parser.add_argument('--dir', required=False, type=str, default='results')
    parser.add_argument('--task', required=False, type=str, default='padding')
    parser.add_argument('--gen_method', required=False, type=str, default='not fix_response')
    parser.add_argument('--filename', required=False, type=str, default='alpaca')
    parser.add_argument('--sample_num', required=False, type=int, default=200)

    args, _ = parser.parse_known_args() 
    main(args)
