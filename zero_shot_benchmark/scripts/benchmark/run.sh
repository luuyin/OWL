shots=0
name_for_output_file=full_llama_30b
model=decapoda-research/llama-30b-hf # model path
python -u run_benchmarking.py \
    --output-path results/task-${shots}-${name_for_output_file}.jsonl \
    --model-name ${model} 
