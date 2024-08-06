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
import torch.nn.functional as F
# from pytorch_transformers import AdamW, WarmupLinearSchedule
from utils.process_utils import *
from utils.function_utils import *
import time

class search_sample():
    def __init__(self, model, tokenizer, ids_dict, lr, loss_dict, mask_dict, all_mask, max_depth, topk) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.all_ids = ids_dict['all_ids']
        self.input_ids = ids_dict['input_ids']
        self.selected_ids = ids_dict['all_ids'].squeeze(0)[1:].long()
        self.start_string_length = ids_dict['start_string_length']
        self.end_string_length = ids_dict['end_string_length']
        self.fix_input_length = ids_dict['fix_input_length']
        self.output_ids = ids_dict['output_ids']

        # self.mask = torch.ones(len(self.input_ids))

        self.input = self.model.model.embed_tokens(ids_dict['all_ids']).squeeze(0)
        self.mask_start = torch.ones(ids_dict['start_string_length'], dtype=torch.float16)
        self.mask_control = torch.ones(ids_dict['end_string_length'], dtype=torch.float16)
        self.mask_end = torch.ones(len(ids_dict['output_ids']), dtype=torch.float16)

        self.lr = lr
        # self.depth = depth
        self.max_depth = max_depth
        self.topk = topk

        self.mask_dict = mask_dict
        self.loss_dict = loss_dict
        self.all_mask = all_mask


    def one_step_gradient(self, mask):
        optimizer = optim.AdamW([mask], lr=self.lr)
        optimizer.zero_grad()
        input = self.model.model.embed_tokens(self.all_ids).squeeze(0)
        # mask_start = torch.ones(self.start_string_length)
        # mask_control = torch.ones(self.end_string_length)
        # mask_end = torch.ones(len(self.output_ids))
        mask_matrix = torch.diag(torch.cat((self.mask_start, mask, self.mask_control, self.mask_end), dim=0)).to('cuda')
        mask_input = torch.matmul(mask_matrix, self.input).unsqueeze(0)
        mask_logits = model_embed(self.model, mask_input, self.fix_input_length, self.selected_ids)
        loss = torch.sum(mask_logits)
        loss.backward(retain_graph=True)

        return mask
    
    def sample_mask(self, potential_ids, mask):

        def unique_tensors(tensor):
            seen = {}
            for i in range(tensor.size(0)):
                # Convert each slice to a tuple for hashing
                view = tensor[i].view(-1)  # Flatten the tensor slice to make it 1D
                tup = tuple(view.tolist())  # Convert to tuple for hashing
                if tup not in seen:
                    seen[tup] = tensor[i]  # Store tensor slice if not seen before

            # Stack all unique slices found
            unique = torch.stack(list(seen.values()))
            return unique
        
        input_sample_list = torch.tensor([]).to('cuda', dtype=torch.float16)
        mask_sample_list = torch.tensor([])
        ids_new_list = torch.tensor([])

        mask_sample = mask.data.clone()
        mask_sample[potential_ids] = 0
        potential_ids_all = torch.where(mask_sample == 0)[0]

        mask_matrix = torch.diag(torch.cat((self.mask_start, mask_sample, self.mask_control, self.mask_end), dim=0)).to('cuda')
        mask_input = torch.matmul(mask_matrix, self.input).unsqueeze(0)

        input_sample_list = torch.cat((input_sample_list, mask_input))
        mask_sample_list = torch.cat((mask_sample_list, mask_sample), dim=0).unsqueeze(0)
        ids_new_list = torch.cat((ids_new_list, potential_ids_all)).unsqueeze(0)

        all_mask_ids = torch.tensor(list(range(len(mask.data))))
        sample_ids = torch.tensor([x for x in all_mask_ids.tolist() if x not in potential_ids_all.tolist()])

        for _ in range(24):
            sample_indices_remove = torch.randperm(len(potential_ids_all))[:1]
            sample_indices_add = torch.randperm(len(sample_ids))[:1]
     
            ids_new = potential_ids_all.clone()
            ids_new[sample_indices_remove] = sample_ids[sample_indices_add]

            mask_sample = torch.ones(len(self.input_ids))
            mask_sample[ids_new] = 0
            mask_matrix = torch.diag(torch.cat((self.mask_start, mask_sample, self.mask_control, self.mask_end), dim=0)).to('cuda', dtype=torch.float16)
            mask_input = torch.matmul(mask_matrix, self.input).unsqueeze(0)
            mask_input = mask_input.to(dtype=torch.float16)
            
            input_sample_list = torch.cat((input_sample_list, mask_input))
            mask_sample_list = torch.cat((mask_sample_list, mask_sample.unsqueeze(0)), dim=0)
            ids_new_list = torch.cat((ids_new_list, ids_new.unsqueeze(0)))

            input_sample_list = unique_tensors(input_sample_list)
            mask_sample_list = unique_tensors(mask_sample_list)
            ids_new_list = unique_tensors(ids_new_list)

            potential_ids_all = torch.where(mask_sample == 0)[0]
            sample_ids = torch.tensor([x for x in all_mask_ids.tolist() if x not in potential_ids_all.tolist()])

        # input_sample_list = input_sample_list.to(dtype=torch.float16)
        mask_logits = torch.tensor([])
        for j in range(0, 24, 4):
            mask_logits1 = model_embed(self.model, input_sample_list[j:(j+4),:,:], self.fix_input_length, self.selected_ids).detach().cpu()
            mask_logits = torch.cat((mask_logits, mask_logits1))

        with torch.no_grad():
            # print(torch.min(torch.sum(mask_logits, -1)!, 0).values)
            min_mask_sample_ids = torch.min(torch.sum(mask_logits, -1), 0).indices
            min_mask_sample = mask_sample_list[min_mask_sample_ids]
            print('sampled tokens:', self.tokenizer.batch_decode(self.input_ids[min_mask_sample==0]))
        
        return ids_new_list, min_mask_sample_ids
    

    def beam_selection_sample(self, mask, depth):
            
        def tensor_in_list(tensor, tensor_list):
            return any(torch.equal(tensor, t) for t in tensor_list)

        if depth == 1:
            # print(mask)
            mask.requires_grad = True
            mask = self.one_step_gradient(mask)
            # print(mask.grad)
            mask_grad = mask.grad.to(dtype=torch.float32)
            _, potential_ids_neg = torch.topk(-mask_grad, k=self.topk, largest=True)
            _, potential_ids_pos = torch.topk(mask_grad, k=self.topk, largest=True)
            potential_ids = torch.cat((potential_ids_neg, potential_ids_pos))
            mask = mask.detach()
            # sample
            # ids_new_list, min_mask_sample_ids = self.sample_mask(potential_ids, mask)
            # potential_ids_new = ids_new_list[min_mask_sample_ids]
            potential_ids_new = potential_ids

            for pid in potential_ids_new:
                # print(pid)
                mask_copy = mask.data.clone()
                mask_copy[int(pid.item())] = 0
                print(f'depth={depth}', self.tokenizer.batch_decode(self.input_ids[mask_copy==0]))
                self.beam_selection_sample(mask_copy, depth+1)

        elif depth < self.max_depth and depth > 1:

            mask_temp = mask.data.clone() # used to calculate the gradient rank
            mask_temp.requires_grad = True
            mask_temp = self.one_step_gradient(mask_temp)
            mask_temp.grad[mask_temp.data==0] = 0
            mask_temp_grad = mask_temp.grad.to(dtype=torch.float32)

            _, potential_ids_neg = torch.topk(-mask_temp_grad, k=self.topk, largest=True)
            _, potential_ids_pos = torch.topk(mask_temp_grad, k=self.topk, largest=True)
            potential_ids = torch.cat((potential_ids_neg, potential_ids_pos))
            mask_temp = mask_temp.detach()

            # ids_new_list, min_mask_sample_ids = self.sample_mask(potential_ids, mask_temp)
            # potential_ids_new = ids_new_list[min_mask_sample_ids]
            potential_ids_new = potential_ids

            loss_list = []
            mask_list = []

            for pid in potential_ids_new:
                mask_copy = mask.data.clone()
                mask_copy[int(pid.item())] = 0

                if tensor_in_list(mask_copy, self.all_mask) == False:
                    self.all_mask.append(mask_copy)
                else:
                    continue

                mask_matrix = torch.diag(torch.cat((self.mask_start, mask_copy, self.mask_control, self.mask_end), dim=0)).to('cuda')
                mask_input = torch.matmul(mask_matrix, self.input).unsqueeze(0)
                mask_logits = model_embed(self.model, mask_input, self.fix_input_length, self.selected_ids)
                loss = torch.sum(mask_logits)
                loss_list.append(loss.item())
                mask_list.append(mask_copy)

            if len(loss_list) > 0:
                loss_min = min(loss_list)
                idx_min = loss_list.index(loss_min)
                mask_copy_final = mask_list[idx_min]
                self.mask_dict[str(depth)].append(mask_copy_final)
                self.loss_dict[str(depth)].append(loss_min)
                print(f'depth={depth}',self.tokenizer.batch_decode(self.input_ids[mask_copy_final==0]))
                self.beam_selection_sample(mask_copy_final, depth+1)

        return self.loss_dict, self.mask_dict


class GSS():
    def __init__(self, data, model, tokenizer, args):
        self.data = data
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.n1 = 0

        self.diff_dict = {'loglikelihood':[], 'tfidf':[], 'bleu':[], 'sentenceBert':[],
                    'rouge1_p':[], 'rouge1_r':[], 'rouge1_f':[],
                    'rouge2_p':[], 'rouge2_r':[], 'rouge2_f':[],
                    'rougel_p':[], 'rougel_r':[], 'rougel_f':[],'kl':[], 'new_log':[]}
        self.diff_dict_keys = self.diff_dict.keys()

        # Creating deep copies for each dictionary
        self.random_token_dict = copy.deepcopy(self.diff_dict)
        self.random_word_dict = copy.deepcopy(self.diff_dict)
        self.token_dict = copy.deepcopy(self.diff_dict)
        self.word_dict = copy.deepcopy(self.diff_dict)
        self.output_all = {}

        self.fix_control1 = 'You are a helpful, respectful and honest assistant. Please analyze the following context and answer the question. '
        self.fix_control2 = ''
        self.fix_control3 = "Of course! I'd be happy to help you with that."

    def cal(self):
        text_save_dir = f"{self.args.dir}/text/{self.args.filename}_{self.args.num}"
        if not os.path.exists(text_save_dir):
            os.mkdir(text_save_dir)

        text_save_path = f"{text_save_dir}/_{self.args.task}_{self.args.num}.json"
        if os.path.exists(text_save_path):
            os.remove(text_save_path)
            print(f"The file {text_save_path} has been deleted.")

        if self.args.filename == 'nq':
            dataset = self.data
            self.data = dataset['org_context']
            self.question = dataset['question']
        elif self.args.filename == 'adversarialqa':
            dataset = self.data
            self.data = dataset['context']
            self.question = dataset['question']

        for i in range(len(self.data)):
            # words_list = data[4].split()
            if self.args.filename in ['nq', 'adversarialqa']:
                self.fix_control2 = self.question[i]

            words_list = self.data[i].split()
            print(i)
            input_sentence = ' '.join(word for word in words_list)
            # fix_string = f'{input_sentence}'
            fix_string = f'{self.fix_control1}{input_sentence}'
            print(f"### Input ### {fix_string}")

            fix_control1_temp = ''
            fix_model_inputs, fix_input_length = tokenizer_query(self.tokenizer, fix_string, fix_control1_temp, self.fix_control2, self.fix_control3, self.args)
            processor = SavingLogitsProcessor()
            fix_generated_ids = self.model.generate(**fix_model_inputs, logits_processor=[processor], max_new_tokens=self.args.max_new_tokens, 
                                            num_beams=1, do_sample=False)
            fix_true_generated_ids = fix_generated_ids[:, fix_input_length:] # delete the indices of the input strings
            if self.args.gen_method == 'fix_response':
                fix_response_ids = fix_true_generated_ids[0][:10]
                fix_response = self.tokenizer.batch_decode([fix_true_generated_ids[0][10:]], skip_special_tokens=True)[0]
            else:
                fix_response = self.tokenizer.batch_decode([fix_true_generated_ids[0]], skip_special_tokens=True)[0]
            fix_probs = processor.prob[self.args.hand_cutting_num:]
            processor.prob = []
            # print(f"probs: {probs}")
            fix_logsum = sum_of_logs(fix_probs)
            print(f"### Output ### {fix_response}")
            print(sum([math.log(x) for x in fix_probs]))

            _, init_prompt_length0 = tokenizer_query(self.tokenizer, '', fix_control1_temp, '', '', self.args) # {Sys}{}{}[/INST]{}
            _, init_prompt_length2 = tokenizer_query(self.tokenizer, '', fix_control1_temp, self.fix_control2, self.fix_control3, self.args) #{Sys}{}{fix_control2}[/INST]{fix_control3}
            inst_length = len(self.tokenizer.encode('[/INST]'))-self.args.hand_cutting_num
            end_string_length = init_prompt_length2 - init_prompt_length0 + inst_length # len(fix_control2)+len(fix_control3)+len([/INST])
            start_string_length = init_prompt_length0 - inst_length # len(Sys)

            all_ids = fix_generated_ids
            start_ids = all_ids.squeeze(0)[:start_string_length][1:]
            input_ids = all_ids.squeeze(0)[start_string_length:(fix_input_length - end_string_length)]
            end_ids = all_ids.squeeze(0)[(fix_input_length-end_string_length):fix_input_length]
            output_ids = all_ids.squeeze(0)[fix_input_length:]

            ids_dict = {'all_ids':all_ids, 'start_ids':start_ids, 'input_ids':input_ids, 'end_ids':end_ids, 'output_ids':output_ids,
                        'start_string_length':start_string_length, 'end_string_length':end_string_length, 'fix_input_length':fix_input_length
                        }

            selected_ids = all_ids.squeeze(0)[1:].long() # all indices except the first one: (x2, x3,..., xn, y1, y2,..., ym, y(m+1),...,y(m+4))
            # token_groups = divide_into_groups(len(input_ids), m=10)  
            # tokens_groups = [list(range(2)), list(range(2,4)), list(range(4, 8)), list(range(8, 16))]
            # groups_num = [2, 4]
            batch_inputs_mask = []
            batch_inputs_mask.append(self.tokenizer.batch_decode(fix_model_inputs['input_ids'][0][1:].unsqueeze(0))[0])

            lr = .1
            mask = torch.ones(len(input_ids)).to(dtype=torch.float16)
            loss_dict = {}; mask_dict = {}; all_mask = []
            
            max_depth = 6
            for d in range(max_depth):
                loss_dict[str(d+1)] = []
                mask_dict[str(d+1)] = []
            current_iteration = 0
            while current_iteration < 1:
                search_sample_ = search_sample(self.model, self.tokenizer, ids_dict, lr, loss_dict, mask_dict, all_mask, max_depth, topk=4)
                loss_dict, mask_dict = search_sample_.beam_selection_sample(mask, 1)
                current_iteration += 1
            
            print('# ------------------------------ Finish Search ------------------------------')
            lr = .1
            mask_tokens_list = []; mask_tokens_list_ids = []
            for k in list(range(1, 6)):
                print(k)
                k = str(k)
                if len(mask_dict[k]) != 0:
                    idx = loss_dict[k].index(min(loss_dict[k]))
                    id_mask = mask_dict[k][idx]
                    idmask = (id_mask==1)
                    input_ids_kept_temp = input_ids[idmask]
                    input_ids_kept = torch.cat((start_ids, input_ids_kept_temp, end_ids), dim=0).unsqueeze(0)
                    new_input_mask = self.tokenizer.batch_decode(input_ids_kept)[0]
                    mask_tokens = self.tokenizer.batch_decode(input_ids[(id_mask==0)])
                    mask_tokens_list.append(mask_tokens)
                    mask_tokens_list_ids.append(input_ids[(id_mask==0)].detach().tolist())
                    print(mask_tokens)
                    print(min(loss_dict[k]))
                    batch_inputs_mask.append(new_input_mask)

            batch_size = len(batch_inputs_mask)
            batch_new_input_ids = self.tokenizer(batch_inputs_mask, padding=True, return_tensors="pt").to("cuda")
            batch_processor = SavingLogitsProcessorBatch(batch_size=batch_size)
            new_batch_probs, new_batch_logsum, new_batch_response, new_batch_generated_ids = model_generate_batch(self.model, self.tokenizer, batch_new_input_ids, batch_processor, self.args)
            batch_processor_change = ChangingSavingLogitsProcessorBatch(aim=fix_true_generated_ids[0], batch_size=batch_size)
            temp_batch_probs, temp_batch_logsum, _, _ = model_generate_batch(self.model, self.tokenizer, batch_new_input_ids, batch_processor_change, self.args)

            print('### New Response ### ')
            for ir, res in enumerate(new_batch_response):
                print(ir)
                print(res)

            text_dict = {}
            text_dict[str(i)] = {'fix_input':fix_string, 'input':input_sentence, 'response': new_batch_response, 'mask_tokens':mask_tokens_list, 'mask_tokens_ids':mask_tokens_list_ids}
            save_dict2json(text_save_path, text_dict)

            token_dict_temp = copy.deepcopy(self.diff_dict)
            # token_dict_temp['new_log'] = []
            # for l in range(batch_size):
            #     processor = ChangingSavingLogitsProcessor(new_batch_generated_ids[l])
            #     generated_ids = self.model.generate(**fix_model_inputs, logits_processor=[processor], max_new_tokens=self.args.max_new_tokens, 
            #                                 num_beams=1, do_sample=False)
            #     new_true_generated_ids = generated_ids[:, fix_input_length:] # delete the indices of the input strings
            #     # fix_response = self.tokenizer.batch_decode(fix_true_generated_ids, skip_special_tokens=True)[0]
            #     new_probs = processor.prob[self.args.hand_cutting_num:]
            #     processor.prob = []
            #     # print(f"probs: {probs}")
            #     new_logsum = sum_of_logs(new_probs)
            #     token_dict_temp['new_log'].append(new_logsum/fix_logsum)

            token_dict_temp = response_sim(fix_response, new_batch_response, diff_dict=token_dict_temp, data_type='batch')
            token_dict_temp['loglikelihood'] = [x/fix_logsum for x in temp_batch_logsum]
            for temp_probs in temp_batch_probs:
                token_dict_temp['kl'].append(kl_divergence([x/sum(fix_probs) for x in fix_probs], [x/sum(temp_probs) for x in temp_probs]))
            token_dict_temp['response'] = new_batch_response

            for key in self.diff_dict_keys:
                self.token_dict[key].append(token_dict_temp[key])

        output_all = {'is_drop': self.token_dict}
        with open(f"{self.args.dir}/output_{self.args.test_model}/search/_{self.args.filename}_{self.args.task}_{self.args.num}.json", 'w') as file:  
            json.dump(output_all, file, indent=4)



class XPrompt():
    def __init__(self, data, model, tokenizer, args):
        self.data = data
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.n1 = 0
        # self.num_start_token = 10
        # self.num_end_token = 4
        self.lr = 0.1

        self.diff_dict = {'loglikelihood':[], 'tfidf':[], 'bleu':[], 'sentenceBert':[],
                    'rouge1_p':[], 'rouge1_r':[], 'rouge1_f':[],
                    'rouge2_p':[], 'rouge2_r':[], 'rouge2_f':[],
                    'rougel_p':[], 'rougel_r':[], 'rougel_f':[],'kl':[], 'new_log':[]}
        self.diff_dict_keys = self.diff_dict.keys()

        # Creating deep copies for each dictionary
        self.token_dict = copy.deepcopy(self.diff_dict)
        self.word_dict = copy.deepcopy(self.diff_dict)
        self.output_all = {}

        # self.fix_control1 = 'You are a helpful, respectful and honest assistant. Please analyze the following context and answer the question. '
        self.fix_control1 = self.args.fix_control1
        self.fix_control2 = self.args.fix_control2
        self.fix_control3 = self.args.fix_control3
        self.use_time_list = []


    def cal(self):
        text_save_dir = f"{self.args.dir}/rebuttal/text/{self.args.filename}_{self.args.num}/"
        if not os.path.exists(text_save_dir):
            os.mkdir(text_save_dir)

        text_save_path = f"{text_save_dir}_{self.args.filename}_{self.args.task}_{self.args.num}_{self.args.test_model}_{self.args.seed}.json"
        if os.path.exists(text_save_path):
            os.remove(text_save_path)
            print(f"The file {text_save_path} has been deleted.")

        # num_list = [2, 3, 4, 5, 10, 15]
        num_list = [2,3,4]
        self.token_dict['loss'] = {}
        for k in num_list:
            self.token_dict['loss'][str(k)] = []
        for i in range(len(self.data)):
            
            words_list = self.data[i].split()
            if len(words_list) == 0:
                continue
            if len(words_list) < 16:
                continue
            print(i)
            input_sentence = ' '.join(word for word in words_list)
            # fix_string = f'{input_sentence}'
            # fix_string = f'{fix_control1}{input_sentence}'
            if self.args.filename not in ['tldr', 'alpaca', 'med']:
                fix_string = f'{self.args.fix_control1}{input_sentence}'
                fix_control1_temp = ''
            elif self.args.filename in ['tldr', 'alpaca', 'med']:
                fix_string = f'{input_sentence}'
                fix_control1_temp = self.args.fix_control1
            print(f"### Input ### {fix_string}")
            # Generate the original response
            # fix_model_inputs, fix_input_length = tokenizer_query(fix_string, tokenizer, args.test_model)
            fix_model_inputs, fix_input_length = tokenizer_query(self.tokenizer, fix_string, fix_control1_temp, self.args.fix_control2, self.args.fix_control3, self.args)
            processor = SavingLogitsProcessor()
            fix_generated_ids = self.model.generate(**fix_model_inputs, logits_processor=[processor], max_new_tokens=self.args.max_new_tokens, 
                                            num_beams=1, do_sample=False)
            fix_true_generated_ids = fix_generated_ids[:, fix_input_length:] # delete the indices of the input strings
            if self.args.gen_method == 'fix_response':
                fix_response_ids = fix_true_generated_ids[0][:10]
                fix_response = self.tokenizer.batch_decode([fix_true_generated_ids[0][10:]], skip_special_tokens=True)[0]
            else:
                fix_response = self.tokenizer.batch_decode([fix_true_generated_ids[0]], skip_special_tokens=True)[0]
                
            fix_probs = processor.prob[self.args.hand_cutting_num:]
            processor.prob = []
            # print(f"probs: {probs}")
            fix_logsum = sum_of_logs(fix_probs)
            fix_loss = sum([math.log(x) for x in fix_probs])
            print(f"### Output ### {fix_response}")
            print(sum([math.log(x) for x in fix_probs]))
            print('# ------------------------------ Input ------------------------------')
            print(self.tokenizer.batch_decode(fix_model_inputs['input_ids'])[0])

            # _, init_prompt_length0 = tokenizer_query(self.tokenizer, '', fix_control1_temp, '', '', self.args) # {Sys}{}{}[/INST]{}
            # _, init_prompt_length2 = tokenizer_query(self.tokenizer, '', fix_control1_temp, self.args.fix_control2, self.args.fix_control3, self.args) #{Sys}{}{fix_control2}[/INST]{fix_control3}
            # inst_length = len(self.tokenizer.encode('[/INST]'))-self.args.hand_cutting_num
            # end_string_length = init_prompt_length2 - init_prompt_length0 + inst_length # len(fix_control2)+len(fix_control3)+len([/INST])
            # start_string_length = init_prompt_length0 - inst_length # len(Sys)

            # all_ids = fix_generated_ids
            # start_ids = all_ids.squeeze(0)[:start_string_length]
            # input_ids = all_ids.squeeze(0)[start_string_length:(fix_input_length - end_string_length)]
            # end_ids = all_ids.squeeze(0)[(fix_input_length-end_string_length):fix_input_length]
            # output_ids = all_ids.squeeze(0)[fix_input_length:]

            # ids_dict = {'all_ids':all_ids, 'start_ids':start_ids, 'input_ids':input_ids, 'end_ids':end_ids, 'output_ids':output_ids,
            #             'start_string_length':start_string_length, 'end_string_length':end_string_length, 'fix_input_length':fix_input_length
            #             }

            if self.args.test_model in ['llama', 'llama_13b']:
            
                _, init_prompt_length0 = tokenizer_query(self.tokenizer, '', fix_control1_temp, '', '', self.args) # {Sys}{}{}[/INST]{}
                _, init_prompt_length2 = tokenizer_query(self.tokenizer, '', fix_control1_temp, self.fix_control2, self.fix_control3, self.args) #{Sys}{}{fix_control2}[/INST]{fix_control3}
                inst_length = len(self.tokenizer.encode('[/INST]'))-self.args.hand_cutting_num
                end_string_length = init_prompt_length2 - init_prompt_length0 + inst_length # len(fix_control2)+len(fix_control3)+len([/INST])
                start_string_length = init_prompt_length0 - inst_length # len(Sys)


                all_ids = fix_generated_ids
                start_ids = all_ids.squeeze(0)[:start_string_length][1:]
                input_ids = all_ids.squeeze(0)[start_string_length:(fix_input_length - end_string_length)]
                end_ids = all_ids.squeeze(0)[(fix_input_length-end_string_length):fix_input_length]
                output_ids = all_ids.squeeze(0)[fix_input_length:]

                ids_dict = {'all_ids':all_ids, 'start_ids':start_ids, 'input_ids':input_ids, 'end_ids':end_ids, 'output_ids':output_ids,
                            'start_string_length':start_string_length, 'end_string_length':end_string_length, 'fix_input_length':fix_input_length
                            }
            elif self.args.test_model == 'vicuna':
                _, init_prompt_length0 = tokenizer_query(self.tokenizer, '', fix_control1_temp, '', '', self.args) # {Sys}{}{}[/INST]{}
                end_string_length = len(self.tokenizer('\n### ASSISTANT:')['input_ids']) - self.args.hand_cutting_num
                start_string_length = init_prompt_length0 - end_string_length

                all_ids = fix_generated_ids
                start_ids = all_ids.squeeze(0)[:start_string_length][1:]
                input_ids = all_ids.squeeze(0)[start_string_length:(fix_input_length - end_string_length)]
                end_ids = all_ids.squeeze(0)[(fix_input_length-end_string_length):fix_input_length]
                output_ids = all_ids.squeeze(0)[fix_input_length:]
                ids_dict = {'all_ids':all_ids, 'start_ids':start_ids, 'input_ids':input_ids, 'end_ids':end_ids, 'output_ids':output_ids,
                            'start_string_length':start_string_length, 'end_string_length':end_string_length, 'fix_input_length':fix_input_length
                            }


            selected_ids = all_ids.squeeze(0)[1:].long() # all indices except the first one: (x2, x3,..., xn, y1, y2,..., ym, y(m+1),...,y(m+4))
            # token_groups = divide_into_groups(len(input_ids), m=10)  
            # tokens_groups = [list(range(2)), list(range(2,4)), list(range(4, 8)), list(range(8, 16))]
            # groups_num = [2, 4]
            batch_inputs_mask = []
            batch_inputs_mask.append(self.tokenizer.batch_decode(fix_model_inputs['input_ids'][0][1:].unsqueeze(0))[0])

            # word_token_groups = token_word_mapping[str(i)]['word_token_list']
            # words_mapping = token_word_mapping[str(i)]['decode_tokens']
            def one_step_gradient(mask):
                optimizer = optim.AdamW([mask], lr=self.lr)
                optimizer.zero_grad()
                input = self.model.model.embed_tokens(all_ids).squeeze(0)
                mask_start = torch.ones(start_string_length); mask_control = torch.ones(end_string_length); mask_end = torch.ones(len(output_ids))
                mask_matrix = torch.diag(torch.cat((mask_start, mask, mask_control, mask_end), dim=0)).to('cuda', dtype=torch.float16)
                mask_input = torch.matmul(mask_matrix, input).unsqueeze(0)
                mask_logits = model_embed(self.model, mask_input, fix_input_length, selected_ids)
                loss = torch.sum(mask_logits)
                loss.backward()
                return mask
            
            def sample_tokens(mask, subset1, subset2, loss):
                prob_list1 = torch.abs(mask.grad)[subset1]
                prob1 = torch.softmax(prob_list1, dim=-1)
                prob_list2 = torch.abs(mask.grad)[subset2]
                prob2 = torch.softmax(prob_list2, dim=-1)

                num_samples = 1  # For example, to draw one sample
                sampled_idx1 = torch.multinomial(prob1, num_samples, replacement=True)
                sampled_idx2 = torch.multinomial(prob2, num_samples, replacement=True)
                subset1_new = subset1.clone()
                subset1_new[sampled_idx1] = subset2[sampled_idx2]
                idmask = torch.ones(input_ids.size(0), dtype=torch.bool)  # Initially set all to True
                idmask[subset1] = False
                subset2_new = torch.tensor(list(range(len(input_ids))))[idmask]

                mask_new = torch.ones(len(input_ids))
                mask_new[subset1_new] = 0

                input = self.model.model.embed_tokens(all_ids).squeeze(0)
                mask_start = torch.ones(start_string_length); mask_control = torch.ones(end_string_length); mask_end = torch.ones(len(output_ids))
                mask_matrix = torch.diag(torch.cat((mask_start, mask_new, mask_control, mask_end), dim=0)).to('cuda', dtype=torch.float16)
                mask_input = torch.matmul(mask_matrix, input).unsqueeze(0)
                mask_logits = model_embed(self.model, mask_input, fix_input_length, selected_ids)
                # loss_new = torch.sum(mask_logits).detach()
                loss_new = sum_of_logs([math.exp(x) for x in mask_logits.detach().tolist()[0]])
                if loss_new < loss:
                    loss = loss_new
                    subset1 = subset1_new.clone()
                    subset2 = subset2_new.clone()
                return subset1, subset2, loss
            
            mask_dict = {}
            loss_dict = {}
        
            use_time_list = []
            for k in num_list:
                loss_dict[str(k)] = []
                mask = torch.ones(len(input_ids))
                lr = .1
                mask.requires_grad = True
                mask = one_step_gradient(mask)
                grad_values, potential_ids = torch.topk(-mask.grad, k=k)

                subset1 = potential_ids.clone()
                idmask = torch.ones(input_ids.size(0), dtype=torch.bool)  # Initially set all to True
                idmask[subset1] = False
                subset2 = torch.tensor(list(range(len(input_ids))))[idmask]

                mask_new = torch.ones(len(input_ids))
                mask_new[subset1] = 0
                input = self.model.model.embed_tokens(all_ids).squeeze(0)
                mask_start = torch.ones(start_string_length); mask_control = torch.ones(end_string_length); mask_end = torch.ones(len(output_ids))
                mask_matrix = torch.diag(torch.cat((mask_start, mask_new, mask_control, mask_end), dim=0)).to('cuda', dtype=torch.float16)
                mask_input = torch.matmul(mask_matrix, input).unsqueeze(0)
                mask_logits = model_embed(self.model, mask_input, fix_input_length, selected_ids)
                # loss = torch.sum(mask_logits).detach()
                loss = sum_of_logs([math.exp(x) for x in mask_logits.detach().tolist()[0]])
                print(f'loss:{loss/fix_logsum}')

                start_time = time.time()
                for _ in range(self.args.sample_num):
                    subset1, subset2, loss = sample_tokens(mask, subset1, subset2, loss)
                    # print(loss)
                    loss_dict[str(k)].append(loss/fix_logsum)
                end_time = time.time()
                print(f'Time {end_time-start_time}')
                use_time_list.append(end_time-start_time)
                
                mask_final = torch.ones(len(input_ids))
                mask_final[subset1] = 0
                mask_dict[str(k)] = mask_final

            self.use_time_list.append(np.mean(use_time_list))
            
            print('# ------------------------------ Finish Search ------------------------------')
            lr = .1
            mask_tokens_list = []; mask_tokens_list_ids = []
            for k in mask_dict.keys():
                id_mask = mask_dict[k]
                idmask = (id_mask==1)
                input_ids_kept_temp = input_ids[idmask]
                input_ids_kept = torch.cat((start_ids[1:], input_ids_kept_temp, end_ids), dim=0).unsqueeze(0)
                new_input_mask = self.tokenizer.batch_decode(input_ids_kept)[0]
                mask_tokens = self.tokenizer.batch_decode(input_ids[(id_mask==0)])
                mask_tokens_list.append(mask_tokens)
                mask_tokens_list_ids.append(input_ids[(id_mask==0)].detach().tolist())
                print(mask_tokens)
                batch_inputs_mask.append(new_input_mask)

            batch_size = len(batch_inputs_mask)
            batch_new_input_ids = self.tokenizer(batch_inputs_mask, padding=True, return_tensors="pt").to("cuda")
            batch_processor = SavingLogitsProcessorBatch(batch_size=batch_size)
            new_batch_probs, new_batch_logsum, new_batch_response, new_batch_generated_ids = model_generate_batch(self.model, self.tokenizer, batch_new_input_ids, batch_processor, self.args)
            batch_processor_change = ChangingSavingLogitsProcessorBatch(aim=fix_true_generated_ids[0], batch_size=batch_size)
            temp_batch_probs, temp_batch_logsum, _, _ = model_generate_batch(self.model, self.tokenizer, batch_new_input_ids, batch_processor_change, self.args)

            print('### New Response ### ')
            for ir, res in enumerate(new_batch_response):
                print(ir)
                print(res)

            text_dict = {}
            text_dict[str(i)] = {'fix_input':fix_string, 'input':input_sentence, 'response': new_batch_response, 'mask_tokens':mask_tokens_list, 'mask_tokens_ids':mask_tokens_list_ids}
            save_dict2json(text_save_path, text_dict)

            token_dict_temp = copy.deepcopy(self.diff_dict)
            # drop_diff_dict_temp['new_log'] = []
            # for l in range(batch_size):
            #     processor = ChangingSavingLogitsProcessor(new_batch_generated_ids[l])
            #     generated_ids = model.generate(**fix_model_inputs, logits_processor=[processor], max_new_tokens=args.max_new_tokens, 
            #                                 num_beams=1, do_sample=False)
            #     new_true_generated_ids = generated_ids[:, fix_input_length:] # delete the indices of the input strings
            #     # fix_response = self.tokenizer.batch_decode(fix_true_generated_ids, skip_special_tokens=True)[0]
            #     new_probs = processor.prob[args.hand_cutting_num:]
            #     processor.prob = []
            #     # print(f"probs: {probs}")
            #     new_logsum = sum_of_logs(new_probs)
            #     drop_diff_dict_temp['new_log'].append(new_logsum/fix_logsum)

            token_dict_temp = response_sim(fix_response, new_batch_response, diff_dict=token_dict_temp, data_type='batch')
            token_dict_temp['loglikelihood'] = [x/fix_logsum for x in temp_batch_logsum]
            for temp_probs in temp_batch_probs:
                token_dict_temp['kl'].append(kl_divergence([x/sum(fix_probs) for x in fix_probs], [x/sum(temp_probs) for x in temp_probs]))
            token_dict_temp['response'] = new_batch_response

            for key in self.diff_dict_keys:
                self.token_dict[key].append(token_dict_temp[key])
            
            for k in num_list:
                self.token_dict['loss'][str(k)].append(loss_dict[str(k)])

        output_all = {'is_drop': self.token_dict}
        with open(f"{self.args.dir}/output_{self.args.test_model}/search/rebuttal/_{self.args.filename}_{self.args.task}_{self.args.num}_{self.args.seed}.json", 'w') as file:  
            json.dump(output_all, file, indent=4)



class Gradient_Sample_init():
    def __init__(self, data, model, tokenizer, args):
        self.data = data
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.n1 = 0
        # self.num_start_token = 10
        # self.num_end_token = 4
        self.lr = 0.1

        self.diff_dict = {'loglikelihood':[], 'tfidf':[], 'bleu':[], 'sentenceBert':[],
                    'rouge1_p':[], 'rouge1_r':[], 'rouge1_f':[],
                    'rouge2_p':[], 'rouge2_r':[], 'rouge2_f':[],
                    'rougel_p':[], 'rougel_r':[], 'rougel_f':[],'kl':[], 'new_log':[]}
        self.diff_dict_keys = self.diff_dict.keys()

        # Creating deep copies for each dictionary
        self.token_dict = copy.deepcopy(self.diff_dict)
        self.word_dict = copy.deepcopy(self.diff_dict)
        self.output_all = {}

        # self.fix_control1 = 'You are a helpful, respectful and honest assistant. Please analyze the following context and answer the question. '
        self.fix_control1 = self.args.fix_control1
        self.fix_control2 = self.args.fix_control2
        self.fix_control3 = self.args.fix_control3


    def cal(self):
        text_save_dir = f"{self.args.dir}/text/{self.args.filename}_{self.args.num}/"
        if not os.path.exists(text_save_dir):
            os.mkdir(text_save_dir)

        text_save_path = f"{text_save_dir}_{self.args.filename}_{self.args.task}_{self.args.num}.json"
        if os.path.exists(text_save_path):
            os.remove(text_save_path)
            print(f"The file {text_save_path} has been deleted.")

        # num_list = [2, 3, 4, 5, 10, 15]
        num_list = [2,3,4]
        self.token_dict['loss'] = {}
        for k in num_list:
            self.token_dict['loss'][str(k)] = []

        for i in range(len(self.data)):
            
            words_list = self.data[i].split()
            if len(words_list) == 0:
                continue
            print(i)
            input_sentence = ' '.join(word for word in words_list)
            # fix_string = f'{input_sentence}'
            # fix_string = f'{fix_control1}{input_sentence}'
            if self.args.filename not in ['tldr', 'alpaca', 'med']:
                fix_string = f'{self.args.fix_control1}{input_sentence}'
                fix_control1_temp = ''
            elif self.args.filename in ['tldr', 'alpaca', 'med']:
                fix_string = f'{input_sentence}'
                fix_control1_temp = self.args.fix_control1
            print(f"### Input ### {fix_string}")
            # Generate the original response
            # fix_model_inputs, fix_input_length = tokenizer_query(fix_string, tokenizer, args.test_model)
            fix_model_inputs, fix_input_length = tokenizer_query(self.tokenizer, fix_string, fix_control1_temp, self.args.fix_control2, self.args.fix_control3, self.args)
            processor = SavingLogitsProcessor()
            fix_generated_ids = self.model.generate(**fix_model_inputs, logits_processor=[processor], max_new_tokens=self.args.max_new_tokens, 
                                            num_beams=1, do_sample=False)
            fix_true_generated_ids = fix_generated_ids[:, fix_input_length:] # delete the indices of the input strings
            if self.args.gen_method == 'fix_response':
                fix_response_ids = fix_true_generated_ids[0][:10]
                fix_response = self.tokenizer.batch_decode([fix_true_generated_ids[0][10:]], skip_special_tokens=True)[0]
            else:
                fix_response = self.tokenizer.batch_decode([fix_true_generated_ids[0]], skip_special_tokens=True)[0]
            fix_probs = processor.prob[self.args.hand_cutting_num:]
            processor.prob = []
            # print(f"probs: {probs}")
            fix_logsum = sum_of_logs(fix_probs)
            fix_loss = sum([math.log(x) for x in fix_probs])
            print(f"### Output ### {fix_response}")
            print(sum([math.log(x) for x in fix_probs]))
            print('# ------------------------------ Input ------------------------------')
            print(self.tokenizer.batch_decode(fix_model_inputs['input_ids'])[0])

            _, init_prompt_length0 = tokenizer_query(self.tokenizer, '', fix_control1_temp, '', '', self.args) # {Sys}{}{}[/INST]{}
            _, init_prompt_length2 = tokenizer_query(self.tokenizer, '', fix_control1_temp, self.args.fix_control2, self.args.fix_control3, self.args) #{Sys}{}{fix_control2}[/INST]{fix_control3}
            inst_length = len(self.tokenizer.encode('[/INST]'))-self.args.hand_cutting_num
            end_string_length = init_prompt_length2 - init_prompt_length0 + inst_length # len(fix_control2)+len(fix_control3)+len([/INST])
            start_string_length = init_prompt_length0 - inst_length # len(Sys)

            all_ids = fix_generated_ids
            start_ids = all_ids.squeeze(0)[:start_string_length]
            input_ids = all_ids.squeeze(0)[start_string_length:(fix_input_length - end_string_length)]
            end_ids = all_ids.squeeze(0)[(fix_input_length-end_string_length):fix_input_length]
            output_ids = all_ids.squeeze(0)[fix_input_length:]

            ids_dict = {'all_ids':all_ids, 'start_ids':start_ids, 'input_ids':input_ids, 'end_ids':end_ids, 'output_ids':output_ids,
                        'start_string_length':start_string_length, 'end_string_length':end_string_length, 'fix_input_length':fix_input_length
                        }


            selected_ids = all_ids.squeeze(0)[1:].long() # all indices except the first one: (x2, x3,..., xn, y1, y2,..., ym, y(m+1),...,y(m+4))
            # token_groups = divide_into_groups(len(input_ids), m=10)  
            # tokens_groups = [list(range(2)), list(range(2,4)), list(range(4, 8)), list(range(8, 16))]
            # groups_num = [2, 4]
            batch_inputs_mask = []
            batch_inputs_mask.append(self.tokenizer.batch_decode(fix_model_inputs['input_ids'][0][1:].unsqueeze(0))[0])

            # word_token_groups = token_word_mapping[str(i)]['word_token_list']
            # words_mapping = token_word_mapping[str(i)]['decode_tokens']
            def one_step_gradient(mask):
                optimizer = optim.AdamW([mask], lr=self.lr)
                optimizer.zero_grad()
                input = self.model.model.embed_tokens(all_ids).squeeze(0)
                mask_start = torch.ones(start_string_length); mask_control = torch.ones(end_string_length); mask_end = torch.ones(len(output_ids))
                mask_matrix = torch.diag(torch.cat((mask_start, mask, mask_control, mask_end), dim=0)).to('cuda', dtype=torch.float16)
                mask_input = torch.matmul(mask_matrix, input).unsqueeze(0)
                mask_logits = model_embed(self.model, mask_input, fix_input_length, selected_ids)
                loss = torch.sum(mask_logits)
                loss.backward()
                return mask
            
            def sample_tokens(mask, subset1, subset2, loss):
                prob_list1 = torch.abs(mask.grad)[subset1]
                prob1 = torch.softmax(prob_list1, dim=-1)
                prob_list2 = torch.abs(mask.grad)[subset2]
                prob2 = torch.softmax(prob_list2, dim=-1)

                num_samples = 1  # For example, to draw one sample
                sampled_idx1 = torch.multinomial(prob1, num_samples, replacement=True)
                sampled_idx2 = torch.multinomial(prob2, num_samples, replacement=True)
                subset1_new = subset1.clone()
                subset1_new[sampled_idx1] = subset2[sampled_idx2]
                idmask = torch.ones(input_ids.size(0), dtype=torch.bool)  # Initially set all to True
                idmask[subset1] = False
                subset2_new = torch.tensor(list(range(len(input_ids))))[idmask]

                mask_new = torch.ones(len(input_ids))
                mask_new[subset1_new] = 0

                input = self.model.model.embed_tokens(all_ids).squeeze(0)
                mask_start = torch.ones(start_string_length); mask_control = torch.ones(end_string_length); mask_end = torch.ones(len(output_ids))
                mask_matrix = torch.diag(torch.cat((mask_start, mask_new, mask_control, mask_end), dim=0)).to('cuda', dtype=torch.float16)
                mask_input = torch.matmul(mask_matrix, input).unsqueeze(0)
                mask_logits = model_embed(self.model, mask_input, fix_input_length, selected_ids)
                # loss_new = torch.sum(mask_logits).detach()
                loss_new = sum_of_logs([math.exp(x) for x in mask_logits.detach().tolist()[0]])
                if loss_new < loss:
                    loss = loss_new
                    subset1 = subset1_new.clone()
                    subset2 = subset2_new.clone()
                return subset1, subset2, loss
            
            mask_dict = {}
            loss_dict = {}
            num_list = [2, 3, 4, 5, 10, 15]
            for k in num_list:
                loss_dict[str(k)] = []
                mask = torch.ones(len(input_ids))
                lr = .1
                mask.requires_grad = True
                mask = one_step_gradient(mask)
                grad_values, potential_ids = torch.topk(torch.abs(mask.grad), k=k)

                # subset1 = potential_ids.clone()
                subset1 = torch.tensor(random.sample(list(range(len(input_ids))), k))
                idmask = torch.ones(input_ids.size(0), dtype=torch.bool)  # Initially set all to True
                idmask[subset1] = False
                subset2 = torch.tensor(list(range(len(input_ids))))[idmask]

                mask_new = torch.ones(len(input_ids))
                mask_new[subset1] = 0
                input = self.model.model.embed_tokens(all_ids).squeeze(0)
                mask_start = torch.ones(start_string_length); mask_control = torch.ones(end_string_length); mask_end = torch.ones(len(output_ids))
                mask_matrix = torch.diag(torch.cat((mask_start, mask_new, mask_control, mask_end), dim=0)).to('cuda', dtype=torch.float16)
                mask_input = torch.matmul(mask_matrix, input).unsqueeze(0)
                mask_logits = model_embed(self.model, mask_input, fix_input_length, selected_ids)
                # loss = torch.sum(mask_logits).detach()
                loss = sum_of_logs([math.exp(x) for x in mask_logits.detach().tolist()[0]])

                start_time = time.time()
                for _ in range(self.args.sample_num):
                    subset1, subset2, loss = sample_tokens(mask, subset1, subset2, loss)
                    loss_dict[str(k)].append(loss/fix_logsum)
                end_time = time.time()
                print(f'Time {end_time-start_time}')
                
                mask_final = torch.ones(len(input_ids))
                mask_final[subset1] = 0
                mask_dict[str(k)] = mask_final
            
            
            print('# ------------------------------ Finish Search ------------------------------')
            lr = .1
            mask_tokens_list = []; mask_tokens_list_ids = []
            for k in mask_dict.keys():
                id_mask = mask_dict[k]
                idmask = (id_mask==1)
                input_ids_kept_temp = input_ids[idmask]
                input_ids_kept = torch.cat((start_ids[1:], input_ids_kept_temp, end_ids), dim=0).unsqueeze(0)
                new_input_mask = self.tokenizer.batch_decode(input_ids_kept)[0]
                mask_tokens = self.tokenizer.batch_decode(input_ids[(id_mask==0)])
                mask_tokens_list.append(mask_tokens)
                mask_tokens_list_ids.append(input_ids[(id_mask==0)].detach().tolist())
                print(mask_tokens)
                batch_inputs_mask.append(new_input_mask)

            batch_size = len(batch_inputs_mask)
            batch_new_input_ids = self.tokenizer(batch_inputs_mask, padding=True, return_tensors="pt").to("cuda")
            batch_processor = SavingLogitsProcessorBatch(batch_size=batch_size)
            new_batch_probs, new_batch_logsum, new_batch_response, new_batch_generated_ids = model_generate_batch(self.model, self.tokenizer, batch_new_input_ids, batch_processor, self.args)
            batch_processor_change = ChangingSavingLogitsProcessorBatch(aim=fix_true_generated_ids[0], batch_size=batch_size)
            temp_batch_probs, temp_batch_logsum, _, _ = model_generate_batch(self.model, self.tokenizer, batch_new_input_ids, batch_processor_change, self.args)

            print('### New Response ### ')
            for ir, res in enumerate(new_batch_response):
                print(ir)
                print(res)

            text_dict = {}
            text_dict[str(i)] = {'fix_input':fix_string, 'input':input_sentence, 'response': new_batch_response, 'mask_tokens':mask_tokens_list, 'mask_tokens_ids':mask_tokens_list_ids}
            save_dict2json(text_save_path, text_dict)

            token_dict_temp = copy.deepcopy(self.diff_dict)
            # drop_diff_dict_temp['new_log'] = []
            # for l in range(batch_size):
            #     processor = ChangingSavingLogitsProcessor(new_batch_generated_ids[l])
            #     generated_ids = model.generate(**fix_model_inputs, logits_processor=[processor], max_new_tokens=args.max_new_tokens, 
            #                                 num_beams=1, do_sample=False)
            #     new_true_generated_ids = generated_ids[:, fix_input_length:] # delete the indices of the input strings
            #     # fix_response = self.tokenizer.batch_decode(fix_true_generated_ids, skip_special_tokens=True)[0]
            #     new_probs = processor.prob[args.hand_cutting_num:]
            #     processor.prob = []
            #     # print(f"probs: {probs}")
            #     new_logsum = sum_of_logs(new_probs)
            #     drop_diff_dict_temp['new_log'].append(new_logsum/fix_logsum)

            token_dict_temp = response_sim(fix_response, new_batch_response, diff_dict=token_dict_temp, data_type='batch')
            token_dict_temp['loglikelihood'] = [x/fix_logsum for x in temp_batch_logsum]
            for temp_probs in temp_batch_probs:
                token_dict_temp['kl'].append(kl_divergence([x/sum(fix_probs) for x in fix_probs], [x/sum(temp_probs) for x in temp_probs]))
            token_dict_temp['response'] = new_batch_response

            for key in self.diff_dict_keys:
                self.token_dict[key].append(token_dict_temp[key])
            for k in num_list:
                self.token_dict['loss'][str(k)].append(loss_dict[str(k)])

        output_all = {'is_drop': self.token_dict}
        with open(f"{self.args.dir}/output_{self.args.test_model}/search/_{self.args.filename}_{self.args.task}_{self.args.num}.json", 'w') as file:  
            json.dump(output_all, file, indent=4)



class Gradient_Sample_prob():
    def __init__(self, data, model, tokenizer, args):
        self.data = data
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.n1 = 0
        # self.num_start_token = 10
        # self.num_end_token = 4
        self.lr = 0.1

        self.diff_dict = {'loglikelihood':[], 'tfidf':[], 'bleu':[], 'sentenceBert':[],
                    'rouge1_p':[], 'rouge1_r':[], 'rouge1_f':[],
                    'rouge2_p':[], 'rouge2_r':[], 'rouge2_f':[],
                    'rougel_p':[], 'rougel_r':[], 'rougel_f':[],'kl':[], 'new_log':[]}
        self.diff_dict_keys = self.diff_dict.keys()

        # Creating deep copies for each dictionary
        self.token_dict = copy.deepcopy(self.diff_dict)
        self.word_dict = copy.deepcopy(self.diff_dict)
        self.output_all = {}

        # self.fix_control1 = 'You are a helpful, respectful and honest assistant. Please analyze the following context and answer the question. '
        self.fix_control1 = self.args.fix_control1
        self.fix_control2 = self.args.fix_control2
        self.fix_control3 = self.args.fix_control3


    def cal(self):
        text_save_dir = f"{self.args.dir}/text/{self.args.filename}_{self.args.num}/"
        if not os.path.exists(text_save_dir):
            os.mkdir(text_save_dir)

        text_save_path = f"{text_save_dir}_{self.args.filename}_{self.args.task}_{self.args.num}.json"
        if os.path.exists(text_save_path):
            os.remove(text_save_path)
            print(f"The file {text_save_path} has been deleted.")

        # num_list = [2, 3, 4, 5, 10, 15]
        num_list = [2,3,4]
        self.token_dict['loss'] = {}
        for k in num_list:
            self.token_dict['loss'][str(k)] = []

        for i in range(len(self.data)):
            
            words_list = self.data[i].split()
            if len(words_list) == 0:
                continue
            print(i)
            input_sentence = ' '.join(word for word in words_list)
            # fix_string = f'{input_sentence}'
            # fix_string = f'{fix_control1}{input_sentence}'
            if self.args.filename not in ['tldr', 'alpaca', 'med']:
                fix_string = f'{self.args.fix_control1}{input_sentence}'
                fix_control1_temp = ''
            elif self.args.filename in ['tldr', 'alpaca', 'med']:
                fix_string = f'{input_sentence}'
                fix_control1_temp = self.args.fix_control1
            print(f"### Input ### {fix_string}")
            # Generate the original response
            # fix_model_inputs, fix_input_length = tokenizer_query(fix_string, tokenizer, args.test_model)
            fix_model_inputs, fix_input_length = tokenizer_query(self.tokenizer, fix_string, fix_control1_temp, self.args.fix_control2, self.args.fix_control3, self.args)
            processor = SavingLogitsProcessor()
            fix_generated_ids = self.model.generate(**fix_model_inputs, logits_processor=[processor], max_new_tokens=self.args.max_new_tokens, 
                                            num_beams=1, do_sample=False)
            fix_true_generated_ids = fix_generated_ids[:, fix_input_length:] # delete the indices of the input strings
            if self.args.gen_method == 'fix_response':
                fix_response_ids = fix_true_generated_ids[0][:10]
                fix_response = self.tokenizer.batch_decode([fix_true_generated_ids[0][10:]], skip_special_tokens=True)[0]
            else:
                fix_response = self.tokenizer.batch_decode([fix_true_generated_ids[0]], skip_special_tokens=True)[0]
            fix_probs = processor.prob[self.args.hand_cutting_num:]
            processor.prob = []
            # print(f"probs: {probs}")
            fix_logsum = sum_of_logs(fix_probs)
            fix_loss = sum([math.log(x) for x in fix_probs])
            print(f"### Output ### {fix_response}")
            print(sum([math.log(x) for x in fix_probs]))
            print('# ------------------------------ Input ------------------------------')
            print(self.tokenizer.batch_decode(fix_model_inputs['input_ids'])[0])

            _, init_prompt_length0 = tokenizer_query(self.tokenizer, '', fix_control1_temp, '', '', self.args) # {Sys}{}{}[/INST]{}
            _, init_prompt_length2 = tokenizer_query(self.tokenizer, '', fix_control1_temp, self.args.fix_control2, self.args.fix_control3, self.args) #{Sys}{}{fix_control2}[/INST]{fix_control3}
            inst_length = len(self.tokenizer.encode('[/INST]'))-self.args.hand_cutting_num
            end_string_length = init_prompt_length2 - init_prompt_length0 + inst_length # len(fix_control2)+len(fix_control3)+len([/INST])
            start_string_length = init_prompt_length0 - inst_length # len(Sys)

            all_ids = fix_generated_ids
            start_ids = all_ids.squeeze(0)[:start_string_length]
            input_ids = all_ids.squeeze(0)[start_string_length:(fix_input_length - end_string_length)]
            end_ids = all_ids.squeeze(0)[(fix_input_length-end_string_length):fix_input_length]
            output_ids = all_ids.squeeze(0)[fix_input_length:]

            ids_dict = {'all_ids':all_ids, 'start_ids':start_ids, 'input_ids':input_ids, 'end_ids':end_ids, 'output_ids':output_ids,
                        'start_string_length':start_string_length, 'end_string_length':end_string_length, 'fix_input_length':fix_input_length
                        }


            selected_ids = all_ids.squeeze(0)[1:].long() # all indices except the first one: (x2, x3,..., xn, y1, y2,..., ym, y(m+1),...,y(m+4))
            # token_groups = divide_into_groups(len(input_ids), m=10)  
            # tokens_groups = [list(range(2)), list(range(2,4)), list(range(4, 8)), list(range(8, 16))]
            # groups_num = [2, 4]
            batch_inputs_mask = []
            batch_inputs_mask.append(self.tokenizer.batch_decode(fix_model_inputs['input_ids'][0][1:].unsqueeze(0))[0])

            # word_token_groups = token_word_mapping[str(i)]['word_token_list']
            # words_mapping = token_word_mapping[str(i)]['decode_tokens']
            def one_step_gradient(mask):
                optimizer = optim.AdamW([mask], lr=self.lr)
                optimizer.zero_grad()
                input = self.model.model.embed_tokens(all_ids).squeeze(0)
                mask_start = torch.ones(start_string_length); mask_control = torch.ones(end_string_length); mask_end = torch.ones(len(output_ids))
                mask_matrix = torch.diag(torch.cat((mask_start, mask, mask_control, mask_end), dim=0)).to('cuda', dtype=torch.float16)
                mask_input = torch.matmul(mask_matrix, input).unsqueeze(0)
                mask_logits = model_embed(self.model, mask_input, fix_input_length, selected_ids)
                loss = torch.sum(mask_logits)
                loss.backward()
                return mask
            
            def sample_tokens(mask, subset1, subset2, loss):
                # prob_list1 = torch.abs(mask.grad)[subset1]
                # prob1 = torch.softmax(prob_list1, dim=-1)
                # prob_list2 = torch.abs(mask.grad)[subset2]
                # prob2 = torch.softmax(prob_list2, dim=-1)
                prob_list1 = torch.ones(subset1.size(0))
                prob1 = prob_list1/prob_list1.sum()
                prob_list2 = torch.ones(subset2.size(0))
                prob2 = prob_list2/prob_list2.sum()

                num_samples = 1  # For example, to draw one sample
                sampled_idx1 = torch.multinomial(prob1, num_samples, replacement=True)
                sampled_idx2 = torch.multinomial(prob2, num_samples, replacement=True)
                subset1_new = subset1.clone()
                subset1_new[sampled_idx1] = subset2[sampled_idx2]
                idmask = torch.ones(input_ids.size(0), dtype=torch.bool)  # Initially set all to True
                idmask[subset1] = False
                subset2_new = torch.tensor(list(range(len(input_ids))))[idmask]

                mask_new = torch.ones(len(input_ids))
                mask_new[subset1_new] = 0

                input = self.model.model.embed_tokens(all_ids).squeeze(0)
                mask_start = torch.ones(start_string_length); mask_control = torch.ones(end_string_length); mask_end = torch.ones(len(output_ids))
                mask_matrix = torch.diag(torch.cat((mask_start, mask_new, mask_control, mask_end), dim=0)).to('cuda', dtype=torch.float16)
                mask_input = torch.matmul(mask_matrix, input).unsqueeze(0)
                mask_logits = model_embed(self.model, mask_input, fix_input_length, selected_ids)
                # loss_new = torch.sum(mask_logits).detach()
                loss_new = sum_of_logs([math.exp(x) for x in mask_logits.detach().tolist()[0]])
                if loss_new < loss:
                    loss = loss_new
                    subset1 = subset1_new.clone()
                    subset2 = subset2_new.clone()
                return subset1, subset2, loss
            
            mask_dict = {}
            loss_dict = {}
            # num_list = [2, 3, 4, 5, 10, 15]
            for k in num_list:
                loss_dict[str(k)] = []
                mask = torch.ones(len(input_ids))
                lr = .1
                mask.requires_grad = True
                mask = one_step_gradient(mask)
                grad_values, potential_ids = torch.topk(torch.abs(mask.grad), k=k)

                # subset1 = potential_ids.clone()
                subset1 = potential_ids.clone()
                idmask = torch.ones(input_ids.size(0), dtype=torch.bool)  # Initially set all to True
                idmask[subset1] = False
                subset2 = torch.tensor(list(range(len(input_ids))))[idmask]

                mask_new = torch.ones(len(input_ids))
                mask_new[subset1] = 0
                input = self.model.model.embed_tokens(all_ids).squeeze(0)
                mask_start = torch.ones(start_string_length); mask_control = torch.ones(end_string_length); mask_end = torch.ones(len(output_ids))
                mask_matrix = torch.diag(torch.cat((mask_start, mask_new, mask_control, mask_end), dim=0)).to('cuda', dtype=torch.float16)
                mask_input = torch.matmul(mask_matrix, input).unsqueeze(0)
                mask_logits = model_embed(self.model, mask_input, fix_input_length, selected_ids)
                # loss = torch.sum(mask_logits).detach()
                # loss = sum_of_logs(mask_logits.detach().tolist())
                loss = sum_of_logs([math.exp(x) for x in mask_logits.detach().tolist()[0]])

                start_time = time.time()
                for _ in range(self.args.sample_num):
                    subset1, subset2, loss = sample_tokens(mask, subset1, subset2, loss)
                    loss_dict[str(k)].append(loss/fix_logsum)
                end_time = time.time()
                print(f'Time {end_time-start_time}')
                
                mask_final = torch.ones(len(input_ids))
                mask_final[subset1] = 0
                mask_dict[str(k)] = mask_final
            
            
            print('# ------------------------------ Finish Search ------------------------------')
            lr = .1
            mask_tokens_list = []; mask_tokens_list_ids = []
            for k in mask_dict.keys():
                id_mask = mask_dict[k]
                idmask = (id_mask==1)
                input_ids_kept_temp = input_ids[idmask]
                input_ids_kept = torch.cat((start_ids[1:], input_ids_kept_temp, end_ids), dim=0).unsqueeze(0)
                new_input_mask = self.tokenizer.batch_decode(input_ids_kept)[0]
                mask_tokens = self.tokenizer.batch_decode(input_ids[(id_mask==0)])
                mask_tokens_list.append(mask_tokens)
                mask_tokens_list_ids.append(input_ids[(id_mask==0)].detach().tolist())
                print(mask_tokens)
                batch_inputs_mask.append(new_input_mask)

            batch_size = len(batch_inputs_mask)
            batch_new_input_ids = self.tokenizer(batch_inputs_mask, padding=True, return_tensors="pt").to("cuda")
            batch_processor = SavingLogitsProcessorBatch(batch_size=batch_size)
            new_batch_probs, new_batch_logsum, new_batch_response, new_batch_generated_ids = model_generate_batch(self.model, self.tokenizer, batch_new_input_ids, batch_processor, self.args)
            batch_processor_change = ChangingSavingLogitsProcessorBatch(aim=fix_true_generated_ids[0], batch_size=batch_size)
            temp_batch_probs, temp_batch_logsum, _, _ = model_generate_batch(self.model, self.tokenizer, batch_new_input_ids, batch_processor_change, self.args)

            print('### New Response ### ')
            for ir, res in enumerate(new_batch_response):
                print(ir)
                print(res)

            text_dict = {}
            text_dict[str(i)] = {'fix_input':fix_string, 'input':input_sentence, 'response': new_batch_response, 'mask_tokens':mask_tokens_list, 'mask_tokens_ids':mask_tokens_list_ids}
            save_dict2json(text_save_path, text_dict)

            token_dict_temp = copy.deepcopy(self.diff_dict)

            token_dict_temp = response_sim(fix_response, new_batch_response, diff_dict=token_dict_temp, data_type='batch')
            token_dict_temp['loglikelihood'] = [x/fix_logsum for x in temp_batch_logsum]
            for temp_probs in temp_batch_probs:
                token_dict_temp['kl'].append(kl_divergence([x/sum(fix_probs) for x in fix_probs], [x/sum(temp_probs) for x in temp_probs]))
            token_dict_temp['response'] = new_batch_response

            for key in self.diff_dict_keys:
                self.token_dict[key].append(token_dict_temp[key])
            for k in num_list:
                self.token_dict['loss'][str(k)].append(loss_dict[str(k)])

        output_all = {'is_drop': self.token_dict}
        with open(f"{self.args.dir}/output_{self.args.test_model}/search/_{self.args.filename}_{self.args.task}_{self.args.num}.json", 'w') as file:  
            json.dump(output_all, file, indent=4)


