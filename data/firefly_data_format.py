# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai 
# @file: firefly_data_format.py
# @time: 2023/8/10 23:38
import json
import jsonlines

with open('spo.json', 'r', encoding='utf-8') as f:
    spo_list = json.loads(f.read())

i = 0
max_length = 0
for content, spos in spo_list.items():
    i += 1
    message = {"conversation_id": i,
               "category": "IE",
               "conversation": [{"human": f'给定以下文本，请分析并提取其中的关系三元组。每个三元组应该包括主体（人物、组织或物体）、'
                                          f'关系和客体（人物、地点或物体）。如果文本中没有明显的关系，请返回空字符串。\n\n'
                                          f'文本： "{content}"\n\n请按照以下格式提取关系三元组列表：\n- （主体，关系，客体）'
                                          f'\n- （主体，关系，客体）\n\n如果没有可识别的关系，请返回空字符串。',
                                 "assistant": '\n'.join([f"（{_[0]}，{_[1]}，{_[2]}）" for _ in spos]) if spos else ''}],
               "dataset": "spo"
               }
    max_length = max(max_length, len(message["conversation"][0]["human"] + message["conversation"][0]["assistant"]))
    print(message)
    with jsonlines.open("spo.jsonl", 'a') as w:
        w.write(message)

print(max_length)
print(i)
