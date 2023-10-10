#  [Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity](https://arxiv.org/abs/2310.05175)

Official PyTorch implementation of  **OWL**: A Missing Secret Sauce for Pruning LLMs to High Sparsity

[Lu Yin](https://luuyin.com//), [You Wu](https://research.google/people/YouWillWu/), [Zhenyu Zhang](https://scholar.google.com/citations?user=ZLyJRxoAAAAJ&hl=zh-CN), [Cheng-Yu Hsieh](https://chengyuhsieh.github.io/), [Yaqing Wang](https://yaqingwang.github.io/), [Yiling Jia](https://yilingjia.github.io/), [Mykola Pechenizkiy](https://www.tue.nl/en/research/researchers/mykola-pechenizkiy), [Yi Liang](https://research.google/people/108265/), [Zhangyang Wang](https://vita-group.github.io/), [Shiwei Liu](https://shiweiliuiiiiiii.github.io/)

**University of Texas at Austin, Eindhoven University of Technology, Google Research, NY. University of Washington.**

The code can be contacted at l.yin@tue.nl.


<p align="center">
<img src="./Images/Layer_wise_sparsity.png" width="700" height="200">
</p>

 OWL layerwise sparsity and Uniform layerwise sparsity at 70% sparsity. The bar chart in background corresponds to the Layerwise Outlier Distribution.



## Table of contents

* [Abstract](#abstract)


* [Results](#Results)

* [Installation](#installation)
* [Usage](#Usage)


## TL;DR
--- 
Inspiring by the strong correlation to the emergent outliers in the feature dimensions in LLMs, we propose effective layer-wise sparsity ratios for LLM pruning, achieving significant improvement.


## Abstract
--- 
Large Language Models (LLMs), renowned for their remarkable performance across diverse domains, present a challenge due to their colossal model size when it comes to practical deployment. In response to this challenge, efforts have been directed toward the application of traditional network pruning techniques to LLMs, uncovering a massive number of parameters can be pruned without hurting performance. Building upon insights gained from pre-LLM models, particularly BERT-level language models, prevailing LLM pruning strategies have consistently adhered to the practice of uniformly pruning all layers at equivalent sparsity levels, resulting in robust performance. However, this observation stands in contrast to the prevailing trends observed in the field of vision models, where non-uniform layerwise sparsity typically yields substantially improved results. To elucidate the underlying reasons for this disparity, we conduct a comprehensive analysis of the distribution of token features within LLMs. In doing so, we discover a strong correlation with the emergence of outliers, defined as features exhibiting significantly greater magnitudes compared to their counterparts in feature dimensions. Inspired by this finding, we introduce a novel LLM pruning methodology that incorporates a tailored set of non-uniform layerwise sparsity ratios specifically designed for LLM pruning, termed as Outlier Weighed Layerwise sparsity **(OWL)**. The sparsity ratio of OWL is directly proportional to the outlier ratio observed within each layer, facilitating a more effective alignment between layerwise weight sparsity and outlier ratios. Our empirical evaluation, conducted across the LLaMA-V1 family and OPT, spanning various benchmarks, demonstrates the distinct advantages offered by OWL over previous methods. For instance, our approach exhibits a remarkable performance gain, surpassing the state-of-the-art Wanda and SparseGPT by 61.22 and 6.80 perplexity at a high sparsity level of 70%, respectively. Code is submitted.


## Results 
--- 

<p align="center">
<img src="./Images/ppl.png" width="700" height="200">
</p>

<p style="text-align: center;"><i>WikiText validation perplexity of OWL applied to SparseGPT and Wanda</i></p>





| **Method** | **Layerwise Sparsity** | **Weight Update** | **LLaMA-V1 7B** | **LLaMA-V1 13B** | **LLaMA-V1 30B** | **LLaMA-V1 65B** | **OPT 6.7B** |
|------------|------------------------|-------------------|----------------|------------------|------------------|------------------|--------------|
| Dense      | -                      | -                 | 5.68           | 5.09             | 4.10             | 4.77             | 10.13        |
| Magnitude  | Uniform                | ❌               | 48419.12       | 84539.45         | 977.73           | 46.89            | 290985.03    |
| Wanda      | Uniform                | ❌               | 85.77          | 55.90            | 17.37            | 15.23            | 162.92       |
| OWL w. Wanda | Non-Uniform             | ❌               | **24.55** | **17.17** | **10.75** | **8.61** | **40.22** |
| SparseGPT  | Uniform                | ✔️               | 26.30          | 19.24            | 12.56            | 10.45            | **20.29**    |
| OWL w. SparseGPT | Non-Uniform         | ✔️               | **19.49** | **14.55** | **10.28** | **8.28** | 22.48   |


<p style="text-align: center;"><i>WikiText validation perplexity of pruning methods for LLaMA-V1 family and OPT-6.7B at 70% sparsity. 
The best performance method is indicated in <b>bold </b>, and the gain in perplexity achieved by OWL is highlighted in blue.</i></p>



## Installation 
--- 
Installation instructions can be found in [INSTALL.md](INSTALL.md).



## Usage

--- 
We provide a quick overview of the arguments:  
- `--model`: The identifier for the LLaMA model on the Hugging Face model hub.
- `--cache_dir`: Directory for loading or storing LLM weights. The default is `llm_weights`.
- `--prune_method`: Pruning methods,namely [`wanda_owl`,`sparsegpt_owl`,`magnitude`, `wanda`, `sparsegpt`].
- `--sparsity_ratio`: Denotes the percentage of weights to be pruned.
- `--save`: Specifies the directory where the result will be stored.
- `--Hyper_m`: Denotes the hyperparameter of `M`.
- `--Lamda`:  Denotes the hyperparameter of `Lamda`.




--- 
### Script example of pruning llama-7b using OWL-wanda

```
python   main.py    \
--model_name_or_path decapoda-research/llama-7b-hf     \
--Lamda 0.08 \
--Hyper_m 5 \
--model decapoda-research/llama-7b-hf     \
--prune_method wanda_owl     \
--sparsity_ratio 0.7 \
--sparsity_type unstructured \
--save save_test/
```


### Script example of pruning llama-7b using OWL-SparseGPT
```
python   main.py    \
--model_name_or_path decapoda-research/llama-7b-hf     \
--Lamda 0.08 \
--Hyper_m 5 \
--model decapoda-research/llama-7b-hf     \
--prune_method sparsegpt_owl     \
--sparsity_ratio 0.7 \
--sparsity_type unstructured \
--save save_test/
```

### Script example of pruning llama-7b using OWL-magnitude
```
python   main.py    \
--model_name_or_path decapoda-research/llama-7b-hf     \
--Lamda 0.08 \
--Hyper_m 5 \
--model decapoda-research/llama-7b-hf     \
--prune_method magnitude_owl      \
--sparsity_ratio 0.7 \
--sparsity_type unstructured \
--save save_test/
```

### Acknowledgement
This repository is build upon the [Wanda](https://github.com/locuslab/wanda) and [SparseGPT](https://github.com/IST-DASLab/sparsegpt) repositories.

**More details coming soon!**

## Citation
if you find this repo is helpful, please cite

```
@article{yin2023owl,
  title={Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity},
  author={Lu Yin, You Wu, Zhenyu Zhang, Cheng-Yu Hsieh, Yaqing Wang, Yiling Jia, Mykola Pechenizkiy, Yi Liang, Zhangyang Wang, Shiwei Liu},
  year={2023}
}
```
