"""
ref: https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/huggingface_api.py
Usage:
python Vicuna/vicuna.py
"""
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.conversation import get_default_conv_template, compute_skip_echo_len
from fastchat.serve.inference import load_model
from loader import create_loader
from tqdm import tqdm
import os
from pdb import set_trace

def rmreturn(s):
    '''Remove all the return symbols.'''
    s = s.replace('\n\n', ' ')
    s = s.replace('\n', ' ')
    return s.strip()


def generate_vicuna_training_data(conversation_data, outfile = 'vicuna-webQA_train.json'):
    '''Convert any dataset to vicuna training data format.'''
    
    with open(dataset_path, "rb") as f:
        dataset_data = json.load(f)
    _, sampler = create_loader(
        dataset_data, params=None, is_train=False
    )
    jsonDict = []
    for i, (Q, A, split, Qcate, Guid, img_posFacts_data, img_negFacts_data, txt_posFacts_data, txt_negFacts_data) in enumerate(sampler):
        if split[0] != 'train':
            continue
        if Qcate[0] != 'text':
            continue
        conv = {}
        conv['id'] = Guid[0]
        conv['conversations'] = []

        prompt = f'Question: {Q}\nEvidence:\n'
        for title, fact in txt_posFacts_data:
            prompt +=  f'{fact}\n'

        human_talk = {}
        human_talk['from'] = "human"
        human_talk['value'] = prompt

        vicuna_talk = {}
        vicuna_talk['from'] = "gpt"
        vicuna_talk['value'] = A

        conv['conversations'].append(human_talk)
        conv['conversations'].append(vicuna_talk)
        jsonDict.append(conv)

    with open(outfile, "w") as f:
        json.dump(jsonDict, f)

@torch.inference_mode()
def vicuna_inference(args):
    '''Inference with vicuna'''
    model, tokenizer = load_model(
        args.model_path,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        debug=args.debug,
    )
    # load validation data from WebQA
    with open(args.dataset_path, "rb") as f:
        dataset_data = json.load(f)
    _, sampler = create_loader(
        dataset_data, params=None, is_train=False
    )
    # Open file
    completed_lines = 0
    if os.path.exists(args.outfile):
        outs = open(args.outfile, 'a', encoding='utf8')
        completed_lines = len(open(args.outfile, 'r').readlines())
    else: # not os.path.exists(args.outfile)
        outs = open(args.outfile, 'a', encoding='utf8')

    pbar = tqdm(total = len(sampler))
    for i, (Q, A, split, Qcate, Guid, img_posFacts_data, img_negFacts_data, txt_posFacts_data, txt_negFacts_data) in enumerate(sampler):
        pbar.update(1)
        # skip completed lines
        if i < completed_lines:
            continue
        if split[0] != 'val':
            continue
        if Qcate[0] != 'text':
            continue
        # auto save for every 500 iteration
        if ((i+1) % 500 == 0):
            outs.close()
            outs = open(args.outfile, 'a', encoding='utf8')
        # Create prompt
        prompt = ''
        if len(txt_posFacts_data) > 0:
            prompt += "Read below facts:\n\n"
            for fact in facts:
                prompt += fact
            prompt += f'and answer this question: {Q} Answer:'
        else:
            prompt = f'Question: {Q} Answer:'
        # Tokenize
        inputs = tokenizer([prompt])
        # Generate
        output_ids = model.generate(
            torch.as_tensor(inputs.input_ids).cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=128,
        )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        # Write file
        outs.write(json.dumps({
                'Guid' : Guid[0],
                'Qcate' : Qcate[0],
                'question': Q[0],
                'answer': A[0][0],
                'output': rmreturn(outputs[len(prompt):])})
                +'\n')
    pbar.close()
    outs.close()
        
@torch.inference_mode()
def vicuna_inference_genRead(args):
    '''Inference with vicuna'''
    model, tokenizer = load_model(
        args.model_path,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        debug=args.debug,
    )
    # load validation data from WebQA
    with open(args.dataset_path, "rb") as f:
        dataset_data = json.load(f)
    _, sampler = create_loader(
        dataset_data, params=None, is_train=False
    )
    # Open file
    completed_lines = 0
    if os.path.exists(args.outfile):
        outs = open(args.outfile, 'a', encoding='utf8')
        completed_lines = len(open(args.outfile, 'r').readlines())
    else: # not os.path.exists(args.outfile)
        outs = open(args.outfile, 'a', encoding='utf8')

    pbar = tqdm(total = len(sampler))
    for i, (Q, A, split, Qcate, Guid, img_posFacts_data, img_negFacts_data, txt_posFacts_data, txt_negFacts_data) in enumerate(sampler):
        pbar.update(1)
        # skip completed lines
        if i < completed_lines:
            continue
        if split[0] != 'val':
            continue
        if Qcate[0] != 'text':
            continue
        # auto save for every 500 iteration
        if ((i+1) % 500 == 0):
            outs.close()
            outs = open(args.outfile, 'a', encoding='utf8')
        # Create prompt
        print(f'Question: {Q}')
        prompt = f'"Provide a background document from Wikipedia to answer the given question. \n\n {Q} \n\n"'
        # Tokenize
        inputs = tokenizer([prompt])
        # Generate
        output_ids = model.generate(
            torch.as_tensor(inputs.input_ids).cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=1024,
        )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        background = rmreturn(outputs[len(prompt):])
        if len(txt_posFacts_data) > 0:
            for fact in facts:
                background += "\n"
                background += fact
        print(f'background: {background}')
        # Create prompt
        prompt = f"Refer to the passages below and answer the following question. \n\n Passage: {background} \n\n Question: {Q} \n\n The answer is"
        # Tokenize
        inputs = tokenizer([prompt])
        # Generate
        output_ids = model.generate(
            torch.as_tensor(inputs.input_ids).cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=128,
        )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        print(f'output: {rmreturn(outputs[len(prompt):])}')
        # set_trace()
        # Write file
        outs.write(json.dumps({
                'Guid' : Guid[0],
                'Qcate' : Qcate[0],
                'question': Q[0],
                'answer': A[0][0],
                'output': rmreturn(outputs[len(prompt):])})
                +'\n')
    pbar.close()
    outs.close()

@torch.inference_mode()
def vicuna_inference_genRead_posFact(args):
    '''Inference with vicuna'''
    model, tokenizer = load_model(
        args.model_path,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        debug=args.debug,
    )
    # load validation data from WebQA
    with open(args.dataset_path, "rb") as f:
        dataset_data = json.load(f)
    _, sampler = create_loader(
        dataset_data, params=None, is_train=False
    )
    # Open file
    completed_lines = 0
    if os.path.exists(args.outfile):
        outs = open(args.outfile, 'a', encoding='utf8')
        completed_lines = len(open(args.outfile, 'r').readlines())
    else: # not os.path.exists(args.outfile)
        outs = open(args.outfile, 'a', encoding='utf8')

    pbar = tqdm(total = len(sampler))
    for i, (Q, A, split, Qcate, Guid, img_posFacts_data, img_negFacts_data, txt_posFacts_data, txt_negFacts_data) in enumerate(sampler):
        pbar.update(1)
        # skip completed lines
        if i < completed_lines:
            continue
        if split[0] != 'val':
            continue
        if Qcate[0] != 'text':
            continue
        # auto save for every 500 iteration
        if ((i+1) % 500 == 0):
            outs.close()
            outs = open(args.outfile, 'a', encoding='utf8')
        # Create prompt
        print(f'Question: {Q}')
        prompt = f'"Provide a background document from Wikipedia to answer the given question. \n\n {Q} \n\n"'
        # Tokenize
        inputs = tokenizer([prompt])
        # Generate
        output_ids = model.generate(
            torch.as_tensor(inputs.input_ids).cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=1024,
        )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        background = rmreturn(outputs[len(prompt):])
        print(f'background: {background}')
        # Create prompt
        prompt = f"Refer to the passage below and answer the following question. \n\n Passage: {background} \n\n Question: {Q} \n\n The answer is"
        # Tokenize
        inputs = tokenizer([prompt])
        # Generate
        output_ids = model.generate(
            torch.as_tensor(inputs.input_ids).cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=128,
        )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        print(f'output: {rmreturn(outputs[len(prompt):])}')
        # set_trace()
        # Write file
        outs.write(json.dumps({
                'Guid' : Guid[0],
                'Qcate' : Qcate[0],
                'question': Q[0],
                'answer': A[0][0],
                'output': rmreturn(outputs[len(prompt):])})
                +'\n')
    pbar.close()
    outs.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="FastChat/vicuna-13b/",
        help="The path to the weights",
    )
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda", "mps"], default="cuda"
    )
    parser.add_argument(
        "--num-gpus", type=str, default="1"
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="The maximum memory per gpu. Use a string like '13Gib'",
    )
    parser.add_argument(
        "--load-8bit", action="store_true", help="Use 8-bit quantization."
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512
    )
    parser.add_argument(
        "--debug", action="store_true"
    )
    parser.add_argument(
        "--message", type=str, default="Hello! Who are you?"
    )
    parser.add_argument(
        "--outfile", type=str, default="vicuna-13B-output.jsonl"
    )
    parser.add_argument(
        "--dataset_path", type=str, default="../WebQA_train_val.json"
    )
    args = parser.parse_args()

    vicuna_inference_genRead_posFact(args)
