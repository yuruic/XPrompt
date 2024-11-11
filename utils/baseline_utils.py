import sys
sys.path.append('..')
import math
import json
import os
import torch
from logitsProcessors import SavingLogitsProcessor, ChangingSavingLogitsProcessor, SavingLogitsProcessorBatch, ChangingSavingLogitsProcessorBatch
import random
import copy
from captum.attr import (FeatureAblation, LLMAttribution, TextTokenInput, TextTemplateInput)
from utils.process_utils import *
from utils.text_scores import *
import time

class integrated_gradient():
    def __init__(self, model, tokenizer, ids_dict):
        self.model = model
        self.tokenizer = tokenizer
        self.all_ids = ids_dict['all_ids']
        self.input_ids = ids_dict['input_ids']
        self.selected_ids = ids_dict['all_ids'].squeeze(0)[1:].long()
        self.start_string_length = ids_dict['start_string_length']
        self.end_string_length = ids_dict['end_string_length']
        self.fix_input_length = ids_dict['fix_input_length']
        self.output_ids = ids_dict['output_ids']

        self.input = self.model.model.embed_tokens(ids_dict['all_ids']).squeeze(0)
        self.mask_start = torch.ones(ids_dict['start_string_length'])
        self.mask_control = torch.ones(ids_dict['end_string_length'])
        self.mask_end = torch.ones(len(ids_dict['output_ids']))

        self.path_integral_steps = 100

    def importance_score(self):
        all_gradients = []
        for i in range(0, self.path_integral_steps):
            path_mask = torch.ones(len(self.input_ids))*((i/self.path_integral_steps))
            path_mask_matrix = torch.diag(torch.cat((self.mask_start, path_mask, self.mask_control, self.mask_end), dim=0)).to('cuda', dtype=torch.float16)

            path_initial_input = torch.matmul(path_mask_matrix, self.input).unsqueeze(0)
            mask_logits = model_embed(self.model, path_initial_input, self.fix_input_length, self.selected_ids)
            path_target_probs = torch.sum(mask_logits)

            gradients = torch.autograd.grad(path_target_probs, path_initial_input, retain_graph=False)[0]
            all_gradients.append(gradients)

        path_integral = torch.sum(torch.cat(all_gradients), 0)
        integrated_gradient = torch.sum(path_integral[None] / (self.path_integral_steps + 1) * self.input, -1)[0]
        logit_importance_score = torch.unsqueeze(integrated_gradient, 0)[:,self.start_string_length : (self.fix_input_length-self.end_string_length)]

        return logit_importance_score
    

class BaselineIntegratedGradient():
    def __init__(self, data, model, tokenizer, args):
        self.data = data
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.n1 = 0
        # self.num_start_token = 10
        # self.num_end_token = 4

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

        self.fix_control1 = self.args.fix_control1
        self.fix_control2 = self.args.fix_control2
        self.fix_control3 = self.args.fix_control3

        self.args.output_file = f'output_{self.args.test_model}'
        self.use_time_list = []

    def cal(self):

        text_save_dir = f"{self.args.dir}/text/{self.args.filename}_{self.args.num}/{self.args.output_file}"
        if not os.path.exists(text_save_dir):
            os.mkdir(text_save_dir)

        text_save_path_ig_token = f"{text_save_dir}/_ig_{self.args.num}_{self.args.seed}.json"
        if os.path.exists(text_save_path_ig_token):
            os.remove(text_save_path_ig_token)
            print(f"The file {text_save_path_ig_token} has been deleted.")

        # if self.args.filename == 'nq':
        #     dataset = self.data
        #     self.data = dataset['org_context']
        #     self.question = dataset['question']
        # elif self.args.filename == 'adversarialqa':
        #     dataset = self.data
        #     self.data = dataset['context']
        #     self.question = dataset['question']
        # words_list = data[4].split()
        for i in range(len(self.data)):
            # if self.args.filename in ['nq', 'adversarialqa']:
            #     self.fix_control2 = self.question[i]
            
            words_list = self.data[i].split()
            if len(words_list)==0:
                continue
            print(i)
            input_sentence = ' '.join(word for word in words_list)
            if self.args.filename not in ['tldr', 'alpaca', 'nq', 'med']:
                fix_string = f'{self.fix_control1}{input_sentence}'
                fix_control1_temp = ''
            elif self.args.filename in ['tldr', 'alpaca', 'nq', 'med']:
                fix_string = f'{input_sentence}'
                fix_control1_temp = self.fix_control1
            print(f"### Input ### {fix_string}")
            fix_model_inputs, fix_input_length = tokenizer_query(self.tokenizer, fix_string, fix_control1_temp,self.fix_control2, self.fix_control3, self.args)
            processor = SavingLogitsProcessor()
            
            # fix_generated_ids = self.model.generate(**fix_model_inputs, logits_processor=[processor], max_new_tokens=self.args.max_new_tokens, 
            #                                 num_beams=1, do_sample=False)
            fix_generated_ids = self.model.generate(**fix_model_inputs, logits_processor=[processor], max_new_tokens=self.args.max_new_tokens, 
                                            num_beams=1, do_sample=False)
            fix_true_generated_ids = fix_generated_ids[:, fix_input_length:] # delete the indices of the input strings
        
            fix_response = self.tokenizer.batch_decode([fix_true_generated_ids[0]], skip_special_tokens=True)[0]
            fix_probs = processor.prob[self.args.hand_cutting_num:]
            processor.prob = []
            # print(f"probs: {probs}")
            fix_logsum = sum_of_logs(fix_probs)
            print(f"### Output ### {fix_response}")
            print(sum([math.log(x) for x in fix_probs]))

            # _, init_prompt_length0 = tokenizer_query(self.tokenizer, '', fix_control1_temp, '', '', self.args) # {Sys}{}{}[/INST]{}
            # _, init_prompt_length2 = tokenizer_query(self.tokenizer, '', fix_control1_temp, self.fix_control2, self.fix_control3, self.args) #{Sys}{}{fix_control2}[/INST]{fix_control3}
            # inst_length = len(self.tokenizer.encode('[/INST]'))-self.args.hand_cutting_num
            # end_string_length = init_prompt_length2 - init_prompt_length0 + inst_length # len(fix_control2)+len(fix_control3)+len([/INST])
            # start_string_length = init_prompt_length0 - inst_length # len(Sys)

            # all_ids = fix_generated_ids
            # start_ids = all_ids.squeeze(0)[:start_string_length][1:]
            # input_ids = all_ids.squeeze(0)[start_string_length:(fix_input_length - end_string_length)]
            # end_ids = all_ids.squeeze(0)[(fix_input_length-end_string_length):fix_input_length]
            # output_ids = all_ids.squeeze(0)[fix_input_length:]

            # ids_dict = {'all_ids':all_ids, 'start_ids':start_ids, 'input_ids':input_ids, 'end_ids':end_ids, 'output_ids':output_ids,
            #             'start_string_length':start_string_length, 'end_string_length':end_string_length, 'fix_input_length':fix_input_length
            #             }
            if self.args.test_model == 'llama':
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

            integrated_gradient_ = integrated_gradient(self.model, self.tokenizer, ids_dict)
            ig_is = integrated_gradient_.importance_score()

            rank_scores_dif, rank_indices = torch.tensor(ig_is).topk(len(input_ids))
            rank_tokens = input_ids[rank_indices[0]]
            # rank_tokens = [input_ids[k] for k in torch.topk(ig_is, k=len(input_ids), largest=True).indices]

            tokens_groups = [list(range(2)), list(range(2,3)), list(range(3,4)), list(range(4,5)), list(range(5,10)), list(range(10, 15)), list(range(15, 30)), list(range(30, 40))]
            print('# ------------------------------ Sequentially Padding Token ------------------------------')
            token_dict_temp = copy.deepcopy(self.diff_dict)
            batch_new_inputs = []
            for k in range(len(tokens_groups)):
                tokens_indices_kept = rank_indices[0][tokens_groups[k][0]:]
                # print(tokens_indices_kept)
                mask_tokens_indices_kept = torch.zeros_like(input_ids, dtype=torch.bool)
                mask_tokens_indices_kept[tokens_indices_kept] = True
                input_ids_kept= input_ids[mask_tokens_indices_kept]
                new_input_ids = torch.cat([start_ids,input_ids_kept, end_ids]).unsqueeze(0)
            
                new_input = self.tokenizer.batch_decode(new_input_ids)[0]
                batch_new_inputs.append(new_input)

            batch_size = len(batch_new_inputs)
            batch_new_input_ids = self.tokenizer(batch_new_inputs, padding=True, return_tensors="pt").to("cuda")
            batch_processor = SavingLogitsProcessorBatch(batch_size=batch_size)
            new_batch_probs, new_batch_logsum, new_batch_response, new_batch_generated_ids = model_generate_batch(self.model, self.tokenizer, 
                                                                                                                batch_new_input_ids, batch_processor, self.args)
            batch_processor_change = ChangingSavingLogitsProcessorBatch(aim=fix_true_generated_ids[0], batch_size=batch_size)
            # batch_processor_change = ChangingSavingLogitsProcessorBatchNew(aim=fix_true_generated_ids[0], batch_size=batch_size)
            temp_batch_probs, temp_batch_logsum, _, _ = model_generate_batch(self.model, self.tokenizer, 
                                                                            batch_new_input_ids, batch_processor_change, self.args)

            token_dict_temp = response_sim(fix_response, new_batch_response, diff_dict=token_dict_temp, data_type='batch')
            token_dict_temp['loglikelihood'] = [a/fix_logsum for a in temp_batch_logsum]
            for temp_probs in temp_batch_probs:
                token_dict_temp['kl'].append(kl_divergence([x/sum(fix_probs) for x in fix_probs], [x/sum(temp_probs) for x in temp_probs]))
            token_dict_temp['response'] = new_batch_response

            text_dict = {}
            text_dict[str(i)] = {'fix_input':fix_string, 'input':input_sentence, 'response': new_batch_response, 
                                'rank_tokens': self.tokenizer.batch_decode(torch.tensor(rank_tokens)), 'rank_tokens_ids':rank_indices.detach().tolist()}
            save_dict2json(text_save_path_ig_token, text_dict)

            print('------------------------------ Save Results ------------------------------')

            token_dict_temp['rank_tokens'] = rank_indices.detach().tolist()

            for key in self.diff_dict_keys:
                self.token_dict[key].append(token_dict_temp[key])
            
            diff_dict_all = {}
            diff_dict_all[str(i)] = {'ig':token_dict_temp}
            file_dir = f'{self.args.dir}/{self.args.output_file}/{self.args.filename}_{self.args.num}'
            if not os.path.exists(file_dir):
                os.mkdir(file_dir)
            file_path = f'{file_dir}/score_{self.args.filename}_{self.args.task}_{self.args.num}_{self.args.seed}_all.json'
            save_dict2json(file_path, diff_dict_all)

        output_all = {'token':self.token_dict}

        with open(f"{self.args.dir}/{self.args.output_file}/baseline/_{self.args.filename}_{self.args.task}_{self.args.num}_{self.args.seed}.json", 'w') as file:  
            json.dump(output_all, file, indent=4)


class captum_token():
    def __init__(self, data, model, tokenizer, args):
        self.data = data
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.n1 = 0
        # self.num_start_token = 10
        # self.num_end_token = 4

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

        # self.fix_control1 = 'You are a helpful, respectful and honest assistant. Please analyze the following context and answer the question. '
        self.fix_control1 = self.args.fix_control1
        self.fix_control2 = self.args.fix_control2
        self.fix_control3 = self.args.fix_control3

        self.args.output_file = f'output_{self.args.test_model}'
        self.use_time_list = []

    def cal(self):

        for i in range(len(self.data)):
            words_list = self.data[i].split()
            if len(words_list) < 0:
                continue
            input_sentence = ' '.join(word for word in words_list)
            if self.args.filename not in ['tldr', 'alpaca', 'nq', 'med']:
                fix_string = f'{self.fix_control1}{input_sentence}'
                fix_control1_temp = ''
            elif self.args.filename in ['tldr', 'alpaca', 'nq', 'med']:
                fix_string = f'{input_sentence}'
                fix_control1_temp = self.fix_control1

            print(f"### Input ### {fix_string}")

            # Generate the original response
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
            fix_logsum = sum_of_logs(fix_probs)
            print(f"### Output ### {fix_response}")
            print(sum([math.log(x) for x in fix_probs]))
            
            if self.args.test_model == 'llama':
                system_prompt = 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.'
                eval_prompt = f'''[INST]<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{fix_control1_temp}{fix_string}{self.fix_control2}[/INST]{self.fix_control3}'''

                _, init_prompt_length0 = tokenizer_query(self.tokenizer, '', fix_control1_temp, '', '', self.args) # {Sys}{}{}[/INST]{}
                _, init_prompt_length2 = tokenizer_query(self.tokenizer, '', fix_control1_temp, self.fix_control2, self.fix_control3, self.args) #{Sys}{}{fix_control2}[/INST]{fix_control3}
                inst_length = len(self.tokenizer.encode('[/INST]'))-self.args.hand_cutting_num
                end_string_length = init_prompt_length2 - init_prompt_length0 + inst_length # len(fix_control2)+len(fix_control3)+len([/INST])
                start_string_length = init_prompt_length0 - inst_length # len(Sys)


                all_ids = fix_generated_ids
                start_ids = all_ids.squeeze(0)[:start_string_length][1:]
                input_ids = all_ids.squeeze(0)[start_string_length:(fix_input_length - end_string_length)]
                end_ids = all_ids.squeeze(0)[(fix_input_length-end_string_length):fix_input_length]
     
            elif self.args.test_model == 'vicuna':
                _, init_prompt_length0 = tokenizer_query(self.tokenizer, '', fix_control1_temp, '', '', self.args) # {Sys}{}{}[/INST]{}
                end_string_length = len(self.tokenizer('\n### ASSISTANT:')['input_ids']) - self.args.hand_cutting_num
                start_string_length = init_prompt_length0 - end_string_length

                all_ids = fix_generated_ids
                start_ids = all_ids.squeeze(0)[:start_string_length][1:]
                input_ids = all_ids.squeeze(0)[start_string_length:(fix_input_length - end_string_length)]
                end_ids = all_ids.squeeze(0)[(fix_input_length-end_string_length):fix_input_length]
                output_ids = all_ids.squeeze(0)[fix_input_length:]

                system_prompt = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.'
                eval_prompt = f'{system_prompt}\n### USER: {self.fix_control1}{fix_string}\n### ASSISTANT:'


            fa = FeatureAblation(self.model)
            llm_attr = LLMAttribution(fa, self.tokenizer)
            inp = TextTokenInput(eval_prompt, self.tokenizer)

            start_time = time.time()
            target = fix_response
            attr_res = llm_attr.attribute(inp, target=target)
            seq_attr = attr_res.seq_attr[start_string_length:(fix_input_length - end_string_length)]
            rank_indices = torch.topk(seq_attr, k=seq_attr.shape[0]).indices
            end_time = time.time()
            use_time = end_time-start_time
            print(f'### Time ### {use_time}')
            self.use_time_list.append(use_time)
            # rank_words = [words_list[idx] for idx in rank_indices]
            rank_tokens = [input_ids[idx].item() for idx in rank_indices]

            print(f'### Rank of Tokens ### {rank_tokens}')

            tokens_groups = [list(range(2)), list(range(2,3)), list(range(3,4)), list(range(4,5)), list(range(5,10)), list(range(10, 15)), list(range(15, 30)), list(range(30, 40))]

            print('# ------------------------------ Dropping based on Captum ------------------------------')
            token_dict_temp = copy.deepcopy(self.diff_dict)

            batch_new_inputs = []
            for k3 in range(len(tokens_groups)):
                # print(k3)
                tokens_indices_kept = rank_indices[tokens_groups[k3][0]:]
                mask_tokens_indices_kept = torch.zeros_like(input_ids, dtype=torch.bool)
                mask_tokens_indices_kept[tokens_indices_kept] = True
                input_ids_kept= input_ids[mask_tokens_indices_kept]
                new_input_ids = torch.cat([start_ids,input_ids_kept, end_ids]).unsqueeze(0)
                if self.args.gen_method == 'fix_response':
                    new_input_ids = torch.cat((new_input_ids[0], fix_response_ids), dim=0).unsqueeze(0)
                new_input = self.tokenizer.batch_decode(new_input_ids)[0]
                batch_new_inputs.append(new_input)

            
            batch_size = len(batch_new_inputs)
            batch_new_input_ids = self.tokenizer(batch_new_inputs, padding=True, return_tensors="pt").to("cuda")
            batch_processor = SavingLogitsProcessorBatch(batch_size=batch_size)
            new_batch_probs, new_batch_logsum, new_batch_response, new_batch_generated_ids = model_generate_batch(self.model, self.tokenizer, 
                                                                                                                    batch_new_input_ids, batch_processor, self.args)
            batch_processor_change = ChangingSavingLogitsProcessorBatch(aim=fix_true_generated_ids[0], batch_size=batch_size)
            temp_batch_probs, temp_batch_logsum, _, _ = model_generate_batch(self.model, self.tokenizer, 
                                                                                batch_new_input_ids, batch_processor_change, self.args)
            
            token_dict_temp = response_sim(fix_response, new_batch_response, diff_dict=token_dict_temp, data_type='batch')
            token_dict_temp['loglikelihood'] = [a/fix_logsum for a in temp_batch_logsum]
            for temp_probs in temp_batch_probs:
                token_dict_temp['kl'].append(kl_divergence([x/sum(fix_probs) for x in fix_probs], [x/sum(temp_probs) for x in temp_probs]))
            token_dict_temp['response'] = new_batch_response
            token_dict_temp['rank_indices'] = rank_indices.detach().cpu().tolist()

            for key in self.diff_dict_keys:
                self.token_dict[key].append(token_dict_temp[key])

            diff_dict_all = {}
            diff_dict_all[str(i)] = {'captum':token_dict_temp}
            file_dir = f'{self.args.dir}/{self.args.output_file}/{self.args.filename}_{self.args.num}'


        output_all = {'captum_drop': self.token_dict}
        output_all_path = f"{self.args.dir}/{self.args.output_file}/baseline/rebuttal/_{self.args.filename}_{self.args.task}_{self.args.num}_parallel_{self.args.chunk}_{self.args.seed}.json"
        with open(output_all_path, 'w') as file:  
            json.dump(output_all, file, indent=4)
        save_dict2json(output_all_path, output_all)
            



