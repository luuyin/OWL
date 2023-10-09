

import os
import pdb
import sys
import json 
import tqdm
import copy
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')

    parser.add_argument('--input-path', type=str, default=None)
    parser.add_argument('--output-path', type=str, default=None)

    # options for loading models
    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    model_name = args.model_name

    config = AutoConfig.from_pretrained(model_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)

    ### Loading models
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=args.cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    for task in ['boolq', 'rte', 'hellaswag', 'winogrande', 'openbookqa', 'arc_easy', 'arc_challenge']:
        input_path = 'data/{}-0.jsonl'.format(task)
        output_path = args.output_path.replace('task', task)

        requests = []
        with open(input_path, 'r') as f:
            for line in f:
                if line.strip() != '':
                    requests.append(json.loads(line))

        results = []
        with torch.no_grad():
            for request in tqdm.tqdm(requests):
                result = {'request': request, 'result': {}}
                prompt = request['prompt']
                input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)

                logits = model(input_ids).logits.log_softmax(dim=-1)

                values, indices = logits.squeeze(0).topk(dim=-1, k=1)
                tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
                
                gold_indices = input_ids[:, 1:] # skip first
                logprobs = [None] + torch.gather(logits, -1, gold_indices.unsqueeze(-1)).squeeze(-1).squeeze(0).detach().cpu().tolist()
                top_logprobs = [None] + [{tokenizer.convert_ids_to_tokens(i.item()): v.item()} for v, i in zip(values.squeeze(-1), indices.squeeze(-1))]
                
                result['result'] = {
                    "choices": [
                        {
                            "text": prompt, 
                            "logprobs": {
                                "tokens": tokens, 
                                "token_logprobs": logprobs, 
                                "top_logprobs": top_logprobs, 
                                "text_offset": []
                            }, 
                            "finish_reason": "length"
                        }
                    ], 
                    "request_time": {
                        "batch_time": 0, 
                        "batch_size": 1}
                }
                
                results.append(result)

        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')