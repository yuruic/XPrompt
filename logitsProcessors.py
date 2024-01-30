import torch
from torch import Tensor
from tokenizers import Tokenizer
from transformers import LogitsProcessor
import torch.nn.functional as F
from captum.attr import FeatureAblation , LLMAttribution, TextTemplateInput
# from captum.attr import TextTemplateFeature #Captum 0.7.0 does not have the attribution: TextTemplateFeature

def tokens_to_ids(tokens, tokenizer):
    token_ids = []
    for token in tokens:
        id = tokenizer.encode(token)[-1]
        token_ids.append(id)


class LocalJailLogitsProcessor(LogitsProcessor):

    def __init__(self, tokenizer, prefix, main_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos = 0
        self.prefix_tokens = prefix
        self.delta = 2.0
        self.tokenizer = tokenizer
        self.main_args = main_args
        self.prepare_prefix()
        if self.main_args.remove_negative:
            self.prepare_negative_tokenid()
        if self.main_args.reverse_negative:
            self.prepare_reverse_negative()

    def refresh(self):
        self.pos = 0

    def set_delta(self, delta):
        self.delta = delta

    def set_prefix(self, prefix):
        self.prefix = prefix

    def prepare_prefix(self):
        if self.prefix_tokens == None:
            self.prefix_id = []
        else:
            token_ids = []
            for token in self.prefix_tokens:
                id = self.tokenizer.encode(token)[-1]
                token_ids.append(id)
            self.prefix_id = token_ids

    def prepare_negative_tokenid(self):
        negative_words = ['sorry', 'Sorry', 'cannot', 'apologize', 'illegal', 'unethical', 'apologies', 'However',
                          'Remember', 'Note', 'but', 'But', 'serious', 'dangerous', 'Again']
        w_ids = []
        for w in negative_words:
            token_id = self.tokenizer.encode(w)[-1]
            w_ids.append(token_id)
        self.negative_word_ids = w_ids

    def prepare_reverse_negative(self):
        negative_mid_words = {'sorry': 'glad', 'cannot': 'can', 'illegal': 'legal', 'unethical': 'ethical',
                              'dangerous': 'good', 'serious': 'good'}
        if 'falcon' in self.main_args.model_name_or_path:
            negative_mid_words = {' sorry': ' glad', ' cannot': ' can', ' illegal': ' legal', ' unethical': ' ethical',
                                  ' dangerous': ' good', ' serious': ' good'}
        negative_start_words = ['Sorry', 'However', 'Remember', 'Note', 'But', 'Again', 'As']
        self.nwid2pwid = {}
        self.nswids = []
        for k, v in negative_mid_words.items():
            nwid = self.tokenizer.encode(k)[-1]
            pwid = self.tokenizer.encode(v)[-1]
            self.nwid2pwid[nwid] = pwid
        for nsw in negative_start_words:
            nswid = self.tokenizer.encode(nsw)[-1]
            self.nswids.append(nswid)
        self.ending_id = 2
        if 'falcon' in self.main_args.model_name_or_path:
            self.ending_id = 11

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        # this is lazy to allow us to colocate on the watermarked model's device
        # if self.rng is None:
        #     self.rng = torch.Generator(device=input_ids.device)

        # # NOTE, it would be nice to get rid of this batch loop, but currently,
        # # the seed and partition operations are not tensor/vectorized, thus
        # # each sequence in the batch needs to be treated separately.
        # batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        # for b_idx in range(input_ids.shape[0]):
        #     greenlist_ids = self._get_greenlist_ids(input_ids[b_idx])
        #     batched_greenlist_ids[b_idx] = greenlist_ids

        # green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

        # scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta)
        if not hasattr(self.main_args, "negative_only"):
            if self.pos < len(self.prefix_id):
                # print('Index for Sure', self.tokenizer.encode('Sure'))
                # for index in self.tokenizer.encode('Sure'):
                #     print("??",self.tokenizer.decode(index))
                token_id = self.prefix_id[self.pos]
                scores[:, token_id] += self.delta * 100
                self.pos += 1
        if self.main_args.remove_negative:
            for nw_id in self.negative_word_ids:
                scores[:, nw_id] -= self.delta * 100
        if self.main_args.reverse_negative:
            top_ids = torch.topk(scores, 1)[1]
            if any([x in top_ids for x in self.nswids]):
                print(f'\nGeneration shutdown by {",".join([self.tokenizer.decode(id) for id in top_ids])}\n')
                scores[:, self.ending_id] += self.delta * 100
                scores[:, self.nswids] -= self.delta * 100
            else:
                for nwid, pwid in self.nwid2pwid.items():
                    if nwid in top_ids:
                        print(f"\nReverse {self.tokenizer.decode(nwid)} to {self.tokenizer.decode(pwid)}\n")
                        scores[:, pwid] = scores[:, nwid] + self.delta * 100
                        scores[:, nwid] -= self.delta * 100

        return scores


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