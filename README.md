# XPrompt

This project aims to explain the generation behavior of model outputs for any given prompt while take the joint
effects of the prompt components into account. Assuming that removing the essential parts of the
prompt would result in a significant variation in the model’s output, we propose the novel objective
function and formulate our task of providing faithful counterfactual explanations for the input prompt
as an optimization problem. To quantify the influence of token combinations in the prompt on the
generations, we incorporate a mask approach for joint prompt attribution. Thus, our goal of extracting
the explanations has been converted to finding the optimal mask of the input prompt. We solve
this problem by a probabilistic search algorithm, equipped with gradient guidance and probabilistic
updaten for efficient exploration in the discrete solution space.

The official repository for [XPrompt:Explaining Large Language Model's Generation via Joint Prompt Attribution](https://arxiv.org/abs/2405.20404)


## Installations
```
git clone https://github.com/yuruic/XPrompt.git
cd XPrompt
pip3 install -r requirements.txt
```
## Get joint attributions
```
python run_baseline_padding.py --task search_token_alg --filename alpaca --test_model llama_7b 
```

## Citation
If you found any part of this code is useful in your research, please consider citing our paper:
```
@misc{chang2024xpromptexplaininglargelanguagemodels,
      title={XPrompt:Explaining Large Language Model's Generation via Joint Prompt Attribution}, 
      author={Yurui Chang and Bochuan Cao and Yujia Wang and Jinghui Chen and Lu Lin},
      year={2024},
      eprint={2405.20404},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.20404}, 
}
```
