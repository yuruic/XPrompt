import torch
from transformers import LogitsProcessor
import torch.nn.functional as F


def tokens_to_ids(tokens, tokenizer):
    token_ids = []
    for token in tokens:
        id = tokenizer.encode(token)[-1]
        token_ids.append(id)


class SavingLogitsProcessor(LogitsProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prob = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        all_prob = F.softmax(scores, dim=-1)
        self.prob.append(torch.max(all_prob).item())
        return scores # Minimally working


class ChangingSavingLogitsProcessor(LogitsProcessor):
    def __init__(self, aim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global no_attacking_scores
        no_attacking_scores = []
        self.aim = aim
        self.counter = 0
        self.prob = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        all_prob = F.softmax(scores, dim=-1)
        aim_index = self.aim[self.counter]
        self.prob.append(all_prob[:, aim_index].item()) # select indexed prob
        fake_scores = torch.zeros_like(scores)
        fake_scores[:, aim_index] = 100
        scores = scores + fake_scores
        self.counter += 1
        return scores # Minimally working
    

class SavingLogitsProcessorBatch(LogitsProcessor):
    def __init__(self, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.prob = [[] for _ in range(self.batch_size)]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        all_prob = F.softmax(scores, dim=-1)
        for i in range(self.batch_size):  
            self.prob[i].append(torch.max(all_prob, dim=-1).values[i].item())
        return scores # Minimally working
    
    
class ChangingSavingLogitsProcessorBatch(LogitsProcessor):
    def __init__(self, aim, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global no_attacking_scores
        no_attacking_scores = []
        self.aim = aim
        self.counter = 0
        self.batch_size = batch_size
        self.prob = [[] for _ in range(self.batch_size)]
        

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        all_prob = F.softmax(scores, dim=-1)
        
        aim_index = self.aim[self.counter]
        
        for i in range(self.batch_size):  
            self.prob[i].append(all_prob[i, aim_index].item()) 

        fake_scores = torch.zeros_like(scores)
        for i in range(self.batch_size):
            fake_scores[i, aim_index] = 100 

        scores = scores + fake_scores 
        self.counter += 1

        return scores # Minimally working
    