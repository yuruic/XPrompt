import math
import json
import numpy as np
import os
import torch
from logitsProcessors import SavingLogitsProcessor, ChangingSavingLogitsProcessor, ChangingSavingLogitsProcessorBatch, SavingLogitsProcessorBatch
import copy
import torch.optim as optim
import torch.nn.functional as F
from utils.process_utils import *
from utils.text_scores import *
import time


class Gradient_Sample():
    def __init__(self, data, model, tokenizer, args):
        self.data = data
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.n1 = 0
        self.lr = 0.1

        self.diff_dict = {'loglikelihood':[], 'tfidf':[], 'bleu':[], 'sentenceBert':[],
                    'rouge1_p':[], 'rouge1_r':[], 'rouge1_f':[],
                    'rouge2_p':[], 'rouge2_r':[], 'rouge2_f':[],
                    'rougel_p':[], 'rougel_r':[], 'rougel_f':[],'kl':[], 'new_log':[],
                    'response':[], 'new_responses':[]}
        self.diff_dict_keys = self.diff_dict.keys()

        # Creating deep copies for each dictionary
        self.token_dict = copy.deepcopy(self.diff_dict)
        self.word_dict = copy.deepcopy(self.diff_dict)
        self.output_all = {}

        self.fix_control1 = self.args.fix_control1
        self.fix_control2 = self.args.fix_control2
        self.fix_control3 = self.args.fix_control3
        self.use_time_list = []


    def cal(self):

        num_list = [2, 3, 4, 5, 10, 15]
        self.token_dict['loss'] = {}
        for k in num_list:
            self.token_dict['loss'][str(k)] = []
        for i in range(len(self.data)):
            words_list = self.data[i].split()
            if len(words_list) < 16:
                continue
            input_sentence = ' '.join(word for word in words_list)
            fix_string = f'{input_sentence}'
            fix_control1_temp = self.args.fix_control1

            # Generate the original response
            fix_model_inputs, fix_input_length = tokenizer_query(self.tokenizer, fix_string, fix_control1_temp, self.args.fix_control2, self.args.fix_control3, self.args)
            processor = SavingLogitsProcessor()
            fix_generated_ids = self.model.generate(**fix_model_inputs, logits_processor=[processor], max_new_tokens=self.args.max_new_tokens, 
                                            num_beams=1, do_sample=False)
            fix_true_generated_ids = fix_generated_ids[:, fix_input_length:] # delete the indices of the input strings
            fix_response = self.tokenizer.batch_decode([fix_true_generated_ids[0]], skip_special_tokens=True)[0]
                
            fix_probs = processor.prob[self.args.hand_cutting_num:]
            processor.prob = []
            fix_logsum = sum_of_logs(fix_probs)
            fix_loss = sum([math.log(x) for x in fix_probs])

            print('# ------------------------------ Data Preparation ------------------------------')

            if self.args.test_model in ['llama_7b', 'llama_13b', 'llama_70b']:
            
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

            elif self.args.test_model == 'vicuna':
                _, init_prompt_length0 = tokenizer_query(self.tokenizer, '', fix_control1_temp, '', '', self.args) # {Sys}{}{}[/INST]{}
                end_string_length = len(self.tokenizer('\n### ASSISTANT:')['input_ids']) - self.args.hand_cutting_num
                start_string_length = init_prompt_length0 - end_string_length

                all_ids = fix_generated_ids
                start_ids = all_ids.squeeze(0)[:start_string_length][1:]
                input_ids = all_ids.squeeze(0)[start_string_length:(fix_input_length - end_string_length)]
                end_ids = all_ids.squeeze(0)[(fix_input_length-end_string_length):fix_input_length]
                output_ids = all_ids.squeeze(0)[fix_input_length:]


            selected_ids = all_ids.squeeze(0)[1:].long() 
            batch_inputs_mask = []
            batch_inputs_mask.append(self.tokenizer.batch_decode(fix_model_inputs['input_ids'][0][1:].unsqueeze(0))[0])

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

                loss = sum_of_logs([math.exp(x) for x in mask_logits.detach().tolist()[0]])
                print(f'loss:{loss/fix_logsum}')

                start_time = time.time()
                for _ in range(self.args.sample_num):
                    subset1, subset2, loss = sample_tokens(mask, subset1, subset2, loss)
                    loss_dict[str(k)].append(loss)
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
            for key in mask_dict.keys():
                id_mask = mask_dict[key]
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

            token_dict_temp = copy.deepcopy(self.diff_dict)
            token_dict_temp = response_sim(fix_response, new_batch_response, diff_dict=token_dict_temp, data_type='batch')
            token_dict_temp['loglikelihood'] = [x/fix_logsum for x in temp_batch_logsum]
            for temp_probs in temp_batch_probs:
                token_dict_temp['kl'].append(kl_divergence([x/sum(fix_probs) for x in fix_probs], [x/sum(temp_probs) for x in temp_probs]))
            token_dict_temp['new_responses'] = new_batch_response
            token_dict_temp['response'] = fix_response

            for key in self.diff_dict_keys:
                self.token_dict[key].append(token_dict_temp[key])
            
            for k in num_list:
                self.token_dict['loss'][str(k)].append(loss_dict[str(k)])

        file_save_path = os.path.join(self.args.dir, self.args.test_model, self.args.filename)
        if not os.path.exists(file_save_path):
            os.makedirs(file_save_path)

        output_all = {'is_drop': self.token_dict}
        with open(os.path.join(file_save_path, f"{self.args.task}_{self.args.num}_{self.args.seed}.json"), 'w') as file:  
            json.dump(output_all, file, indent=4)

