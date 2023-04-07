import streamlit as st
from inference import *

with st.sidebar:
    add_select_prob_threshold=st.slider(
        '滑动以选择用于定位span的概率阈值',
        min_value=0.0,max_value=1.0
    )
    add_select_max_seq_len=st.slider(
        '滑动以选择最大句长',
        min_value=0,max_value=512
    )

content=[st.text_input('请输入文本内容，如“2022年11月11日二十条出台。”')]
prompt=[st.text_input('请输入要提取的标签，如“日期”。如返回结果为None，请调低左侧的概率阈值。')]
check=st.checkbox('确定')
if check:
    result=inference(model,tokenizer,'cpu',content,prompt,
                     max_seq_len=add_select_max_seq_len,
                     prob_threshold=add_select_prob_threshold,
                     return_prob=False)
    st.write(f'{prompt[0]}：',result)