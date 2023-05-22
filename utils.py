import streamlit as st
from typing import List
import pandas as pd

# 跨模块调用函数时，被调用函数内部不能使用全局变量。所以需将全局变量定义在源模块内，如从a.py调用b.py中的全局变量c，则需在b.py中定义c
metric_df=pd.DataFrame()
def printMetric(model_name:str,metric:dict) -> None:
    global metric_df
    metric_df[model_name]=metric # 将指标添加到df中

    # metric_col_list内包含指标数量个st.column，metric_col调用metric方法来显示该指标
    metric_col_list=st.columns(len(metric))
    for idx ,metric_col in enumerate(metric_col_list):
        if metric_df.shape[1]>1:
            metric_col.metric(
                metric_df.index[idx],
                metric_df.iloc[idx,-1],
                f'相较于{metric_df.iloc[:,-2].name}{(metric_df.iloc[idx,-1]-metric_df.iloc[idx,-2])/metric_df.iloc[idx,-2]:.3%}'#相较于前一个指标变化了多少百分比
            )
        else:
            metric_col.metric(
                metric_df.index[idx],
                metric_df.iloc[idx,-1],
            )

def selectHyperparameters(model_name,param_type_range_help:List[dict]) -> dict:
    '''Example:
        input: [{'name':'alpha','type':float,'help':'正则化参数','range':[0.1,1.0]},
                {'name':'fit_intercept','type':str,'help':'是否拟合截距','range':[True,False]}]
        output: {'alpha':value,'fit_intercept':True}
    '''
    kwargs={}
    for item in param_type_range_help:
        if item['type']==float:
            kwargs[item['name']]=st.slider(
                label=model_name+' '+item['name'],help=item['help'],
                min_value=item['range'][0],max_value=item['range'][1],
            )
        elif item['type']==int:
            kwargs[item['name']]=st.slider(
                label=model_name+' '+item['name'],help=item['help'],
                min_value=item['range'][0],max_value=item['range'][1],
            )
        elif item['type']==str:
            kwargs[item['name']]=st.selectbox(
                label=model_name+' '+item['name'],help=item['help'],
                options=item['range'],
            )
        else:
            st.warning(body=f"{model_name+item['name']}的类型不符合要求！请从int、float与str三者中择其一。")
    return kwargs