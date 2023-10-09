

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






