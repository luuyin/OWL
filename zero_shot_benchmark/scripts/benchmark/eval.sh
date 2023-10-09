shots=0
name_for_output_file=full_llama_30b
model=decapoda-research/llama-7b-hf
model_arch=llama
for task in boolq rte hellaswag winogrande openbookqa arc_easy arc_challenge
do
python -u evaluate_task_result.py \
    --result-file results/${task}-${shots}-${name_for_output_file}.jsonl \
    --task-name ${task} \
    --num-fewshot ${shots} \
    --model-type ${model_arch}
done
