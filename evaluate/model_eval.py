# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: model_eval.py
# @time: 2023/9/29 23:18
import json

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import torch

# place it into Firefly script dir
import sys
sys.path.append("../../..")
from component.utils import ModelUtils


def extract_spoes(text):
    text = text.strip()
    text = f'给定以下文本，请分析并提取其中的关系三元组。每个三元组应该包括主体（人物、组织或物体）、' \
           f'关系和客体（人物、地点或物体）。如果文本中没有明显的关系，请返回空字符串。\n\n' \
           f'文本： "{text}"\n\n请按照以下格式提取关系三元组列表：\n- （主体，关系，客体）' \
           f'\n- （主体，关系，客体）\n\n如果没有可识别的关系，请返回空字符串。'
    text = f'<s>{text}</s>'

    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    bos_token_id = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)
    eos_token_id = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long).to(device)
    input_ids = torch.concat([bos_token_id, input_ids, eos_token_id], dim=1)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
            top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id
        )
    outputs = outputs.tolist()[0][len(input_ids[0]):]
    response = tokenizer.decode(outputs)
    response = response.strip().replace(tokenizer.eos_token, "").strip()
    return [_[1:-1].split('，') for _ in response.split('\n')]


class SPO(tuple):
    def __init__(self, spo):
        self.spox = tuple(spo)

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


def evaluate(data):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f1, precision, recall = 0, 0, 0
    f = open('pred.json', 'w', encoding='utf-8')
    pbar = tqdm()
    for d in data:
        R = set([SPO(spo) for spo in extract_spoes(d['text'])])
        T = set([SPO(spo) for spo in d['spo_list']])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )
        s = json.dumps({
            'text': d['text'],
            'spo_list_true': list(T),
            'spo_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R)},
            ensure_ascii=False,
            indent=4)
        f.write(s + '\n')
    pbar.close()
    f.close()
    return f1, precision, recall


if __name__ == '__main__':
    # 使用合并后的模型进行推理
    model_name_or_path = '/workspace/Firefly/script/checkpoint/firefly-baichuan2-13b-spo-merge'
    adapter_name_or_path = None

    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = False
    # 生成超参配置
    max_new_tokens = 150
    top_p = 0.9
    temperature = 0.01
    repetition_penalty = 1.0
    device = 'cuda:0'
    # # 加载模型
    model = ModelUtils.load_model(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        adapter_name_or_path=adapter_name_or_path
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == 'llama' else True
    )

    df = pd.read_excel("evaluate_data.xlsx")
    pred_data = []
    for i, row in df.fillna('').iloc[:100].iterrows():
        print(i, row['文本'], row['真实三元组'])
        pred_text, tuple_string = row['文本'], row['真实三元组']
        true_tuples = [_[1:-1].split('，') for _ in tuple_string.split('\n')]
        pred_data.append({'text': pred_text, 'spo_list': true_tuples})

    from pprint import pprint
    pprint(pred_data)

    evaluate(pred_data)
