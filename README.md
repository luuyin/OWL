#  [Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity]()

Official PyTorch implementation of  **OWL**: A Missing Secret Sauce for Pruning LLMs to High Sparsity

[Lu Yin](https://luuyin.com//), [You Wu](https://research.google/people/YouWillWu/), [Zhenyu Zhang](https://scholar.google.com/citations?user=ZLyJRxoAAAAJ&hl=zh-CN), [Cheng-Yu Hsieh](https://chengyuhsieh.github.io/), [Yaqing Wang](https://yaqingwang.github.io/), [Yiling Jia](https://yilingjia.github.io/), [Mykola Pechenizkiy](https://www.tue.nl/en/research/researchers/mykola-pechenizkiy), [Yi Liang](https://research.google/people/108265/), [Zhangyang Wang](https://vita-group.github.io/), [Shiwei Liu](https://shiweiliuiiiiiii.github.io/)

*University of Texas at Austin, Eindhoven University of Technology, Google Research, NY. University of Washington.*

The code can be contacted at l.yin@tue.nl.


<p align="center">
<img src="./Images/Layer_wise_sparsity.png" width="700" height="200">
</p>


<p style="text-align: center;"><i> OWL layerwise sparsity and Uniform layerwise sparsity at 70% sparsity. The bar chart in background corresponds to the Layerwise Outlier Distribution (LOD)<i></p>


<p align="center">
<img src="./Images/ppl.png" width="700" height="200">
</p>

<p style="text-align: center;"><i>WikiText validation perplexity of OWL applied to SparseGPT and Wanda<i></p>





| **Method** | **Layerwise Sparsity** | **Weight Update** | **LLaMA-V1 7B** | **LLaMA-V1 13B** | **LLaMA-V1 30B** | **LLaMA-V1 65B** | **OPT 6.7B** |
|------------|------------------------|-------------------|----------------|------------------|------------------|------------------|--------------|
| Dense      | -                      | -                 | 5.68           | 5.09             | 4.10             | 4.77             | 10.13        |
| Magnitude  | Uniform                | ❌               | 48419.12       | 84539.45         | 977.73           | 46.89            | 290985.03    |
| Wanda      | Uniform                | ❌               | 85.77          | 55.90            | 17.37            | 15.23            | 162.92       |
| OWL w. Wanda | Non-Uni             | ❌               | **24.55 (-61.22)** | **17.17 (-38.73)** | **10.75 (-6.62)** | **8.61 (-6.62)** | **40.22 (-120.70)** |
| SparseGPT  | Uniform                | ✔️               | 26.30          | 19.24            | 12.56            | 10.45            | **20.29**    |
| OWL w. SparseGPT | Non-Uni         | ✔️               | **19.49 (-6.81)** | **14.55 (-4.69)** | **10.28 (-2.28)** | **8.28 (-0.64)** | 22.48 (2.19)  |

<p style="text-align: center;"><i>  WikiText validation perplexity of pruning methods for LLaMA-V1 family and OPT-6.7B at 70% sparsity. 
The best performance method is indicated in <b>bold </b>, and the gain in perplexity achieved by OWL is highlighted in blue.<i></p>




Table of contents
* [Installation](#installation)
* [Usage](#Usage)

--- 

## Installation 
Installation instructions can be found in [INSTALL.md](INSTALL.md).



## Usage

--- 
We provide a quick overview of the arguments:  
- `--model`: The identifier for the LLaMA model on the Hugging Face model hub.
- `--cache_dir`: Directory for loading or storing LLM weights. The default is `llm_weights`.
- `--prune_method`: Pruning methods,namely [`wanda_owl`,`sparsegtp_owl`,`magnitude`, `wanda`, `sparsegpt`].
- `--sparsity_ratio`: Denotes the percentage of weights to be pruned.
- `--save`: Specifies the directory where the result will be stored.
- `--Hyper_m`: Denotes the hyperparameter of `M`.
- `--Lamda`:  Denotes the hyperparameter of `Lamda`.




--- 
### Script example of pruning llama-7b using OWL-wanda

```
for sparsity_ratio in 0.7 
do
    for Lamda in 0.08
    do
        for Hyper_m in 5
        do
             python   main.py    \
            --model_name_or_path decapoda-research/llama-7b-hf     \
            --Lamda $Lamda \
            --Hyper_m $Hyper_m \
            --model decapoda-research/llama-7b-hf     \
            --prune_method wanda_owl     \
            --sparsity_ratio $sparsity_ratio \
            --sparsity_type unstructured \
            --save test/
        done
    done
done

```




## Citation
if you find this repo is helpful, please cite

```

```
