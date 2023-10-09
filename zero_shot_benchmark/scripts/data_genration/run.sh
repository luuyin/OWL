#### simple template
shots=0
for task in boolq rte hellaswag winogrande openbookqa arc_easy arc_challenge; do
python -u generate_task_data.py --output-file ${task}-${shots}.jsonl --task-name ${task} --num-fewshot ${shots}
done


