
import torch
import torch.nn as nn
from transformers import AutoTokenizer,default_data_collator, get_scheduler
from pprint import pprint
from datasets import load_dataset
import json
from functools import partial
import numpy as np
from torch.utils.data import DataLoader
import time
import os
import pandas as pd
from typing import List



# 模型    
device=torch.device('cpu')#在mac上使用metal performances shaders作为pytorch的GPU加速训练后端
model=torch.load('uie-base-zh/pytorch_model.bin',map_location=device)
tokenizer=AutoTokenizer.from_pretrained('uie-base-zh')

# 根据model返回的prob张量中选取span
def get_span(start_ids, end_ids, with_prob=False):
    """
    从get_span函数也能看出来，UIE是想一次性生成多个预测的，比如当给定prompt为“地点”与“时间”，text为“北京”与“2021年”
    Get span set from position start and end list.
    Args:
        start_ids (List[int]/List[tuple]): The start index list.
        end_ids (List[int]/List[tuple]): The end index list.
        with_prob (bool): If True, each element for start_ids and end_ids is a tuple aslike: (index, probability).
    Returns:
        set: The span set without overlapping, every id can only be used once.
    """
    if with_prob:
        start_ids = sorted(start_ids, key=lambda x: x[0])
        end_ids = sorted(end_ids, key=lambda x: x[0])
    else:
        start_ids = sorted(start_ids)
        end_ids = sorted(end_ids)

    start_pointer = 0
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    couple_dict = {}

    # 将每一个span的首/尾token的id进行配对（就近匹配，默认没有overlap的情况）
    while start_pointer < len_start and end_pointer < len_end:
        if with_prob:
            start_id = start_ids[start_pointer][0]
            end_id = end_ids[end_pointer][0]
        else:
            start_id = start_ids[start_pointer]
            end_id = end_ids[end_pointer]

        if start_id == end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            end_pointer += 1
            continue

        if start_id < end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            continue

        if start_id > end_id:
            end_pointer += 1
            continue

    result = [(couple_dict[end], end) for end in couple_dict]
    result = set(result)
    return result
def get_bool_ids_greater_than(probs, limit=0.5, return_prob=False):
    """
    Get idx of the last dimension in probability arrays, which is greater than a limitation.
    最后一维，指的就是从max_seq_len个数字中，找出大于阈值的数字，即概率大于阈值的start_index或end_index
    Args:
        probs (List[List[float]]): The input probability arrays.
        limit (float): The limitation for probability.
        return_prob (bool): Whether to return the probability
    Returns:
        List[List[int]]: The index of the last dimension meet the conditions.
    """
    probs = np.array(probs)
    dim_len = len(probs.shape)
    if dim_len > 1:
        result = []
        for p in probs:
            result.append(get_bool_ids_greater_than(p, limit, return_prob))
        return result
    else:#只有当p为一维张亮时，才会进入该else语句
        result = []
        for i, p in enumerate(probs):
            if p > limit:
                if return_prob:
                    result.append((i, p))
                else:
                    result.append(i)
        return result
    

# 将prompt与content拼接，分词——添加pos_ids——重设offsetmapping
def convertInputs(tokenizer,prompts:List[str],contents:List[str],max_seq_len):
    inputs=tokenizer(
        text=prompts,
        text_pair=contents,
        truncation=True,
        max_length=max_seq_len,
        padding='max_length',
        return_offsets_mapping=True
    )#返回个字典

    pos_ids=[]
    for i in range(len(contents)):
        pos_ids.append([j for j in range(max_seq_len)])
    # pos_ids=torch.tensor(pos_ids)#tensor返回整型，Tensor返回浮点型
    inputs['pos_id']=pos_ids

    #inputs中的mapping，如（0,1）是元组，必须修改为列表才能更改其中元素
    offset_mappings=[[list(mapping) for mapping in offset_mapping] for offset_mapping in inputs['offset_mapping']]
    for i in range(len(offset_mappings)):
        bias=0
        for idx in range(1,len(offset_mappings[i])):#每句话的offset_mapping中，idx为0的一定为prompt的cls，故不用管。
            mapping=offset_mappings[i][idx]
            if mapping[0]==0 and mapping[1]==0 and bias==0:#表示遇到了第二个cls
                bias=offset_mappings[i][idx - 1][1]#源代码中是bias = offset_mappings[i][idx - 1][1]，而验证函数中是idx，这是为什么？
            if mapping[0]==0 and mapping[1]==0:#表示遇到了被padding字符所表示的mapping，即max_sep_len>句长的位置被padding了，其mapping为(0,0)
                continue
            offset_mappings[i][idx][0]+=bias
            offset_mappings[i][idx][1]+=bias
    inputs['offset_mapping']=offset_mappings

    for k,v in inputs.items():
        inputs[k]=torch.LongTensor(v)
    return inputs

#推理
def inference(model,tokenizer,device,contents,prompts,max_seq_len=128,prob_threshold=0.3,return_prob=False):

    inputs=convertInputs(tokenizer,prompts,contents,max_seq_len)

    model.to(device)
    start_prob,end_prob=model(
        input_ids=inputs['input_ids'].to(device),
        token_type_ids=inputs['token_type_ids'].to(device),
        attention_mask=inputs['attention_mask'].to(device)
    )

    pred_start_ids=get_bool_ids_greater_than(start_prob.detach(),limit=prob_threshold,return_prob=return_prob)
    pred_end_ids=get_bool_ids_greater_than(end_prob.detach(),limit=prob_threshold,return_prob=return_prob)

    for pred_start_id,pred_end_id,prompt,content,offset_mapping in zip(pred_start_ids,pred_end_ids,prompts,contents,inputs['offset_mapping']):
        span_set=get_span(pred_start_id,pred_end_id,with_prob=return_prob)#集合不可以切片，但可以迭代
        offset_mapping=offset_mapping.tolist()
        for span in span_set:
            input_content=prompt+content
            span_text=input_content[offset_mapping[span[0]][0]:offset_mapping[span[1]][1]]#要从input_content中提取的内容，称为text
            # print(span_text)
            return span_text
        

if __name__=='__main__':
    content=['2022年11月11日二十条出台。']
    prompt=list('日期')
    print(inference(model,tokenizer,'cpu',content,prompt,max_seq_len=128,prob_threshold=0.3,return_prob=False))