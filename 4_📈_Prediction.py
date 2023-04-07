'''
ideas:
1. st.metric显示指标的变化

'''

import pandas as pd
import numpy as np
import streamlit as st

# ML相关库
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.model_selection import KFold,GridSearchCV,train_test_split,RandomizedSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error

# DL相关库
import torch
import torch.nn as nn
import torch.nn.functional as F

# 作图
from pylab import mpl,plt

# 自定义类
from predict import ML_predict,DL_predict

# 用户选择数据范围、指定模型
with st.sidebar:
    add_select_model=st.multiselect(
        '请选择一个或多个预测模型',
        ['线性回归','支持向量机','LSTM','Transformer']
    )
    st.write('您选择了',add_select_model)

    add_select_region=st.selectbox(
        '请选择一个预测的区域',
        st.session_state['data'].columns
    )
    st.write('您选择了',add_select_region)

    add_select_time=st.slider(
        '请选择数据的起止日期',
        value=(st.session_state['data'].index[0].to_pydatetime(),st.session_state['data'].index[-1].to_pydatetime()),
        format='YYYY/MM/DD',
    )
    st.write('您选择了',add_select_region)

    # 将所选数据标准化
    data=st.session_state['data'].loc[add_select_time[0]:add_select_time[1]]
    scaler=StandardScaler()
    data_pred=scaler.fit_transform(data)
    data_pred=pd.DataFrame(data_pred,index=data.index,columns=data.columns)


# 根据用户所选数据与模型进行预测并可视化
st.subheader('数据已标准化')
metric_df=pd.DataFrame()
for model_name in add_select_model:
    if model_name=='线性回归':
        model=LinearRegression()
        lr=ML_predict(data_pred[add_select_region],model,5,1,0.3)
        metric=lr()#lr()返回指标组成的字典
        metric_df[model_name]=metric#将指标添加到df中
        
        #metric_col_list内包含指标数量个st.column，metric_col调用metric方法来显示该指标
        metric_col_list=st.columns(len(metric))
        for idx,metric_col in enumerate(metric_col_list):
            if metric_df.shape[1]>1:
                metric_col.metric(
                    metric_df.index[idx],#metric_df['线性回归'].name是'线性回归'
                    metric_df.iloc[idx,-1],
                    f'相较于{metric_df.iloc[:,-2].name}{(metric_df.iloc[idx,-1]-metric_df.iloc[idx,-2])/metric_df.iloc[idx,-2]:.3%}'#相较于前一个指标变化了多少百分比
                )
            else:
                metric_col.metric(
                    metric_df.index[idx],#metric_df['线性回归'].name是'线性回归'
                    metric_df.iloc[idx,-1],
                )
    if model_name=='支持向量机':
        model=SVR(kernel='poly')
        lr=ML_predict(data_pred[add_select_region],model,5,1,0.3)
        metric=lr()#lr()返回指标组成的字典
        metric_df[model_name]=metric#将指标添加到df中
        
        #metric_col_list内包含指标数量个st.column，metric_col调用metric方法来显示该指标
        metric_col_list=st.columns(len(metric))
        for idx,metric_col in enumerate(metric_col_list):
            if metric_df.shape[1]>1:
                metric_col.metric(
                    metric_df.index[idx],#metric_df['线性回归'].name是'线性回归'
                    metric_df.iloc[idx,-1],
                    f'相较于{metric_df.iloc[:,-2].name}{(metric_df.iloc[idx,-1]-metric_df.iloc[idx,-2])/metric_df.iloc[idx,-2]:.3%}'#相较于前一个指标变化了多少百分比
                )
            else:
                metric_col.metric(
                    metric_df.index[idx],#metric_df['线性回归'].name是'线性回归'
                    metric_df.iloc[idx,-1],
                )
