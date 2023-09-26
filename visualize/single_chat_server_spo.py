# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: single_chat_server_spo.py
# @time: 2023/7/25 22:27
import os
import json
import gradio as gr
from uuid import uuid4
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def predict(document):
    while '\n\n' in document.strip():
        document = document.replace('\n\n', '\n')
    paras = document.split('\n')
    values = []
    for i, original_text in enumerate(paras):
        original_text = original_text.strip()
        text = f'给定以下文本，请分析并提取其中的关系三元组。每个三元组应该包括主体（人物、组织或物体）、'\
               f'关系和客体（人物、地点或物体）。如果文本中没有明显的关系，请返回空字符串。\n\n'\
               f'文本： "{original_text}"\n\n请按照以下格式提取关系三元组列表：\n- （主体，关系，客体）'\
               f'\n- （主体，关系，客体）\n\n如果没有可识别的关系，请返回空字符串。'
        text = '<s>{}</s>'.format(text)
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id
            )
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = tokenizer.decode(outputs)
        response = response.strip().replace(text, "").replace('</s>', "").replace('<s>', "").strip()
        print('response: ', response)
        values.append([i, original_text, response.replace('\n', '<br>')])
    return values


def update(df):
    samples = []
    for i in range(df.shape[0]):
        no, para, spo = df.iloc[i, :].tolist()
        try:
            if not spo:
                spo_dict = {para: []}
            else:
                spo_dict = {para: [_[1:-1].split('，') for _ in spo.split('<br>')]}
            samples.append(spo_dict)
        except Exception:
            pass
    with open(f'./doc_test/{uuid4()}.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(samples, ensure_ascii=False, indent=4) + '\n')
    return f'write {i+1} records.'


def clear(data):
    return ''


def kg_visualize(df):
    html_dir = "html"
    # remove html
    for file in os.listdir(html_dir):
        os.remove(os.path.join(html_dir, file))
    # handle spo
    outputs = []
    for i in range(df.shape[0]):
        no, para, spo_str = df.iloc[i, :].tolist()
        for line in spo_str.split('<br>'):
            try:
                spo = line[1:-1].split('，')
                outputs.append(
                    json.dumps({"source": spo[0], "target": spo[2], 'rela': spo[1], 'type': "resolved"},
                               ensure_ascii=False))
            except Exception:
                pass
    output_str = ',\n'.join(outputs).replace('"', "'")
    # write and show new html
    with open("index.html", "r") as f:
        content = f.readlines()
    content.insert(9, output_str)
    html_name = str(uuid4())
    with open(f"{html_dir}/{html_name}.html", "w") as g:
        g.writelines(content)
    output_str = f"""<iframe src="file={html_dir}/{html_name}.html" width="100%" height="600px"></iframe>"""
    return output_str


if __name__ == '__main__':
    model_name = '~/Firefly/script/checkpoint/firefly-baichuan-7b-spo-merge'
    max_new_tokens = 150
    top_p = 0.9
    temperature = 0.01
    repetition_penalty = 1.0
    device = 'cuda:0'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == 'llama' else True
    )
    print("model loaded!")

    with gr.Blocks() as demo:
        # 设置输入组件
        doc = gr.Textbox(label="Document", lines=5)
        # 设置输出组件
        output = gr.DataFrame(label='Predict_Result',
                              headers=["no", "para", "spo"],
                              datatype=["number", "str", "str"],
                              interactive=True,
                              wrap=True)
        # 设置按钮
        predict_btn = gr.Button("预测")
        # 设置按钮点击事件
        predict_btn.click(fn=predict, inputs=doc, outputs=output)
        # new button
        result = gr.Textbox(label="Results", lines=1)
        submit_btn = gr.Button("提交")
        submit_btn.click(fn=update, inputs=output, outputs=result)
        # clear button
        clear_btn = gr.ClearButton(value="清除")
        clear_btn.click(fn=clear, inputs=doc, outputs=doc)
        # kg visualize
        kg_output = gr.HTML(label="Kg_visualize")
        kg_btn = gr.Button("图谱可视化")
        kg_btn.click(fn=kg_visualize, inputs=output, outputs=kg_output)

    demo.launch(server_name='0.0.0.0', server_port=7800, share=True)
