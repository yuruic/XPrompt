import argparse
import sys
import math
import random
import json
import numpy as np
import os
import torch
from baselines import compute_bm25_scores, compute_tfidf_scores, compute_sentence_bert_scores

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList)

from logitsProcessors import SavingLogitsProcessor, ChangingSavingLogitsProcessor


vicuna_list = [[0, 10, 17, 19, 20, 21, 29, 31, 33, 40, 43, 45, 48, 51, 53], [2, 5, 7, 9, 11, 12, 14, 16, 24, 26, 27, 34, 36, 38, 39]]
# vicuna_list = [[1, 10, 17, 19, 20, 21], [2, 5, 7, 9, 11, 12, 14]]
llama_list = [[13,32,34,35,36], [2,3,11,18,37], [5,6,7,8,9]]


from defaults import _test_prefixes



_MODELS = {
    "llama": ["/data/bochuan/llm-attacks/DIR/llama/Llama-2-7b-chat-hf", "llama", llama_list],
    "vicuna": ["/data/bochuan/llm-attacks/DIR/vicuna/vicuna-7b-v1.3", "vicuna", vicuna_list],
    "guanaco": ["/data/bochuan/llm-attacks/DIR/guanaco-7B-HF", "guanaco"]
}





def build_chat_input(query, tokenizer, test_model):
    if test_model == "llama":
        input_string = f'''[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.\n<</SYS>>\n\n{query}[/INST]'''
        model_inputs = tokenizer([input_string], truncation=False, return_tensors='pt').to("cuda")
    else:
        input_string = f'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.\n### USER: {query}\n### ASSISTANT:'
        model_inputs = tokenizer([input_string], padding=True,truncation=False, return_tensors='pt').to("cuda")

    # print(model_inputs['input_ids'].shape)
    input_length = model_inputs['input_ids'].shape[-1]
    return model_inputs, input_length

def sum_of_logs(lst):
    return math.exp(sum(math.log(x) for x in lst) / len(lst))

def testing(args, model, tokenizer, adv_dict, all_results, test_model, marker):
    scores = []

    def generate(args, input, processor):
        model_inputs, input_length = build_chat_input(input, tokenizer, test_model)
        generated_ids = model.generate(**model_inputs, logits_processor=[processor], max_new_tokens=args.max_new_tokens)
        true_generated_ids = generated_ids[:, input_length:]
        response = tokenizer.batch_decode(true_generated_ids, skip_special_tokens=True)[0]
        probs = processor.prob[args.hand_cutting_num:]
        processor.prob = []
        print(f"probs: {probs}")
        log_sum = sum_of_logs(probs)
        print(f"attacking score log sum: {log_sum}")
        return response, true_generated_ids, probs, log_sum



    for i in range(len(adv_dict["goal"])):
        print(f"====================================== {marker} case {i} =============================================")
        print(f"------------------------------- with adv prompt -------------------------------------")
        atk_string = f'{adv_dict["goal"][i]} {adv_dict["controls"][i]}'
        processor = SavingLogitsProcessor()
        response, true_atk_generated_ids, attacking_scores, atk_log_sum = generate(args, atk_string, processor)

        if any([prefix in response for prefix in _test_prefixes]):
            print(f"case {i}: existing prefix words, skip")
            continue

        # no attacking
        print(f"------------------------------- without adv prompt -------------------------------------")
        no_atk_string = f'{adv_dict["goal"][i]}'
        processor = ChangingSavingLogitsProcessor(aim=true_atk_generated_ids[0])
        _, true_no_atk_generated_ids, no_attacking_scores, no_atk_log_sum = generate(args, no_atk_string, processor)




        print(f"### goal ###: {adv_dict['goal'][i]}")
        print(f"### control ###: {adv_dict['controls'][i]}")
        print("### response ###:", response)


        split_response = tokenizer.batch_decode(true_atk_generated_ids.reshape(-1,1), skip_special_tokens=True)

        tfidf_score = compute_tfidf_scores([response], adv_dict["controls"][i])[0]
        bm25_score = compute_bm25_scores([response], adv_dict["controls"][i])[0]
        sentenceBert_score = compute_sentence_bert_scores([response], adv_dict["controls"][i])[0]


        scores.append([atk_log_sum, no_atk_log_sum, atk_log_sum / no_atk_log_sum, tfidf_score, bm25_score, sentenceBert_score])

        all_results["index"].append(i)
        all_results["prob"].append([attacking_scores, no_attacking_scores])
        all_results["split_response"].append(split_response)
        all_results["response"].append(response)
        all_results["goal"].append(adv_dict["goal"][i])
        all_results["controls"].append(adv_dict["controls"][i])

    return scores, all_results


def main(args):
    # loading data
    with open(f"./full_results/individual_behavior_controls_full_{args.test_model}.json", 'r') as f:
        log = json.load(f)
    # with open(f"/data/bochuan/llm-attacks/experiments/results/individual_behavior_controls.json", 'r') as f:
    #     log = json.load(f)

    all_dict = {'goal':[], 'controls':[]}



    for i in range(len(log["goal"])):
        all_dict['goal'].append(log['goal'][i])
        all_dict['controls'].append(log['controls'][i])


    # for i in range(len(log["goal"])): print(i, log["controls"][i])

    # loading model
    model_dir = _MODELS[args.test_model][0]
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = 'left'

    all_results = {"index": [], "prob": [], "response": [], "goal": [], "controls": [], "split_response": []}
    all_scores, all_results = testing(args, model, tokenizer, all_dict, all_results, args.test_model, "all_results")



    # save results
    with open(f"./results/all_both_align_results_{args.test_model}_maxNewTokens{args.max_new_tokens}.json", "w") as f:
        json.dump(all_results, f)

    all_scores = np.array(all_scores)
    np.set_printoptions(suppress=True, precision=4)
    print("positive sum(log(prob)) list, \n with adv prompt | without adv prompt | without / with | TF-IDF | BM25 | Sentence Bert\n", all_scores)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='llm metrics')
    parser.add_argument('--hand_cutting_num', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--test_model', type=str, default='vicuna')
    parser.add_argument('--manual_seed', default=0, type=int)
    parser.add_argument('--device', default='3', type=str)
    args = parser.parse_args()


    os.environ["CUDA_VISIBLE_DEVICES"] = args.device


    pos_list = _MODELS[args.test_model][2][0]
    neg_list = _MODELS[args.test_model][2][1]

    main(args)

