'''
ideas:
暂无
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
from torch.optim import AdamW
from model import TimeSeriesTransformer
from transformers import TimeSeriesTransformerConfig,TimeSeriesTransformerForPrediction,InformerConfig,InformerForPrediction
from transformers import PretrainedConfig

# 时间序列相关库
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import time_features_from_frequency_str,get_lags_for_frequency,TimeFeature
from gluonts.transform import (
    AddAgeFeature,AddTimeFeatures,VstackFeatures,
    AddObservedValuesIndicator,
    Chain,
    RenameFields,RemoveFields,SelectFields,
    InstanceSplitter,
)
from gluonts.transform.sampler import ExpectedNumInstanceSampler,ValidationSplitSampler,TestSplitSampler
from gluonts.itertools import Cyclic, IterableSlice, PseudoShuffled
from gluonts.torch.util import IterableDataset

# 作图
from pylab import mpl,plt

# 自定义类
from predict import ML_predict,DL_predict,TST_predict,Informer_predict
from utils import printMetric,selectHyperparameters

col1,col2=st.columns(2)
# 选择模型
with col1:
    with st.form('users choose models'):
        add_select_model=st.multiselect(
            '请选择一个或多个预测模型',
            ['线性回归','支持向量机','Transformer','🤗 Time Series Transformer','🤗 Informer']
        )
        submitted1 = st.form_submit_button('确认')
# 选择数据范围
with col2:
    with st.form('users choose data ranges'):
        add_select_region=st.multiselect(
            '请选择一个预测的区域',
            st.session_state['data'].columns
        )
        add_select_time=st.slider(
            '请选择数据的起止日期',
            value=(st.session_state['data'].index[0].to_pydatetime(),st.session_state['data'].index[-1].to_pydatetime()),
            format='YYYY/MM/DD',
        )
        submitted2 = st.form_submit_button('确认')

if submitted1 or submitted2:
    # 将所选数据标准化
    data=st.session_state['data'].loc[add_select_time[0]:add_select_time[1]]
    scaler=StandardScaler()
    data_pred=scaler.fit_transform(data)
    data_pred=pd.DataFrame(data_pred,index=data.index,columns=data.columns)

    # 向用户展示数据
    col1,col2,col3=st.columns(3)
    with col1:
        st.write('您所选择的模型是：')
        for idx in range(len(add_select_model)):
            st.write(add_select_model[idx])
    with col2:
        st.write('您所选择的区域是：')
        for idx in range(len(add_select_region)):
            st.write(add_select_region[idx])
    with col3:
        st.write('您所选择的起止日期是：')
        for idx in range(len(add_select_time)):
            st.write(add_select_time[idx])
    st.line_chart(data_pred[add_select_region])


# 选择超参数
with st.sidebar:
    st.header('模型的超参数选择')
    for model_name in add_select_model:
        if model_name=='线性回归':
            st.subheader(model_name)
            lr_kwargs=selectHyperparameters(
                model_name,
                [{'name':'fit_intercept','type':str,'help':'拟合截距','range':[True,False]},]
            )
            st.write(f'{model_name}的超参数：')
            st.write(lr_kwargs)

        if model_name=='支持向量机':
            st.subheader(model_name)
            svr_kwargs=selectHyperparameters(
                model_name,
                [{'name':'kernel','type':str,'help':'SVM的核函数','range':['poly','linear', 'rbf', 'sigmoid', 'precomputed']},
                 {'name':'degree','type':int,'help':'二项核的系数','range':[1,5]},
                 {'name':'tol','type':float,'help':'Tolerance for stopping criterion','range':[1e-4,1e-2]},
                 {'name':'C','type':float,'help':'正则化系数','range':[0.1,5.0]},]
            )
            st.write(f'{model_name}的超参数：')
            st.write(svr_kwargs)
        
        if model_name=='🤗 Time Series Transformer':
            st.subheader(model_name)
            tst_kwargs=selectHyperparameters(
                model_name,
                [{'name':'prediction_length','type':int,'help':'传入encoder的序列长度','range':[10,100]},
                 {'name':'context_length','type':int,'help':'传入decoder的序列长度','range':[5,50]},
                 {'name':'n_encoder_layers','type':int,'help':'encoder的层数','range':[1,10]},
                 {'name':'n_decoder_layers','type':int,'help':'decoder的层数','range':[1,10]},
                 {'name':'d_model','type':int,'help':'日后补充','range':[4,32]},
                 {'name':'train_dataloader_batch_size','type':int,'help':'训练集的batch size','range':[16,256]},
                 {'name':'test_dataloader_batch_size','type':int,'help':'测试集的batch size','range':[8,128]},
                 {'name':'epochs','type':int,'help':'训练轮数','range':[1,100]},
                 {'name':'logging_steps','type':int,'help':'每训练指定个数的batch，打印loss','range':[10,100]},]
            )
            st.write(f'{model_name}的超参数：')
            st.write(tst_kwargs)

        if model_name=='🤗 Informer':
            st.subheader(model_name)
            ifm_kwargs=selectHyperparameters(
                model_name,
                [{'name':'prediction_length','type':int,'help':'传入encoder的序列长度','range':[10,100]},
                 {'name':'context_length','type':int,'help':'传入decoder的序列长度','range':[5,50]},
                 {'name':'n_encoder_layers','type':int,'help':'encoder的层数','range':[1,10]},
                 {'name':'n_decoder_layers','type':int,'help':'decoder的层数','range':[1,10]},
                 {'name':'d_model','type':int,'help':'日后补充','range':[4,32]},
                 {'name':'train_dataloader_batch_size','type':int,'help':'训练集的batch size','range':[16,256]},
                 {'name':'test_dataloader_batch_size','type':int,'help':'测试集的batch size','range':[8,128]},
                 {'name':'epochs','type':int,'help':'训练轮数','range':[1,100]},
                 {'name':'logging_steps','type':int,'help':'每训练指定个数的batch，打印loss','range':[10,100]},]
            )
            st.write(f'{model_name}的超参数：')
            st.write(ifm_kwargs)

# 根据用户所选数据与模型进行预测并可视化
if submitted1 or submitted2:
    st.caption('数据已标准化')
    for model_name in add_select_model:
        if model_name=='线性回归':
            model=LinearRegression(**lr_kwargs)
            lr=ML_predict(data_pred[add_select_region],5,1)
            metric=lr(model,0.3)#lr()返回指标组成的字典
            printMetric(model_name,metric)

        if model_name=='支持向量机':
            model=SVR(**svr_kwargs)
            svr=ML_predict(data_pred[add_select_region],5,1)
            metric=svr(model,0.3)#lr()返回指标组成的字典
            printMetric(model_name,metric)

        if model_name=='Transformer':
            model=TimeSeriesTransformer(nvars=1,d_model=512,d_hid=128,nheads=8,nlayers=2,dropout=0.1)
            tst=DL_predict(data_pred[add_select_region],5,1)
            metric=tst(model,0.3,4)
            printMetric(model_name,metric)

        if model_name=='🤗 Time Series Transformer':

            # hyperparameters
            lags_seq=[1,2,3]
            freq='1D'
            num_dynamic_real_features=0
            num_static_categorical_features=0
            num_static_real_features=0
            time_features=time_features_from_frequency_str(freq)

            # define the model
            n_vars=data_pred[add_select_region].shape[1]
            config=TimeSeriesTransformerConfig(
                input_size=n_vars,
                prediction_length=tst_kwargs['prediction_length'],
                context_length=tst_kwargs['context_length'],
                lags_sequence=lags_seq,
                num_time_features=len(time_features)+1,#time_features_from_frequency_str(freq)=3 and add another time feature called AgeFeature
                num_static_categorical_features=num_static_categorical_features,
                cardinality=[len(data_pred[add_select_region])],
                embedding_dimension=[3],

                encoder_layers=tst_kwargs['n_encoder_layers'],
                decoder_layers=tst_kwargs['n_decoder_layers'],
                d_model=tst_kwargs['d_model'],
            )
            model=TimeSeriesTransformerForPrediction(config)

            # train and forcast
            tst=TST_predict(data_pred[add_select_region],tst_kwargs['context_length'],tst_kwargs['prediction_length'])
            tst.gen_src_tgt(freq,config,tst_kwargs['train_dataloader_batch_size'],tst_kwargs['test_dataloader_batch_size'])
            forecast_array,loss_history=tst.predict(model,epochs=tst_kwargs['epochs'],logging_steps=tst_kwargs['logging_steps'])
            metric=tst.evaluate(forecast_array)

            # 可视化各个区域的预测结果
            tab_list=st.tabs(add_select_region)
            for idx,tab in enumerate(tab_list):
                with tab:
                    st.subheader(f'{add_select_region[idx]}的预测结果')
                    tst.plot(freq,idx,forecast_array)
            printMetric(model_name,metric)

        if model_name=='🤗 Informer':

            # hyperparameters
            lags_seq=[1,2,3]
            freq='1D'
            num_dynamic_real_features=0
            num_static_categorical_features=0
            num_static_real_features=0
            time_features=time_features_from_frequency_str(freq)

            # define the model
            n_vars=data_pred[add_select_region].shape[1]
            config=InformerConfig(
                input_size=n_vars,
                prediction_length=ifm_kwargs['prediction_length'],
                context_length=ifm_kwargs['context_length'],
                lags_sequence=lags_seq,
                num_time_features=len(time_features)+1,#time_features_from_frequency_str(freq)=3 and add another time feature called AgeFeature
                num_static_categorical_features=num_static_categorical_features,
                cardinality=[len(data_pred[add_select_region])],
                embedding_dimension=[3],

                encoder_layers=ifm_kwargs['n_encoder_layers'],
                decoder_layers=ifm_kwargs['n_decoder_layers'],
                d_model=ifm_kwargs['d_model'],
            )
            model=InformerForPrediction(config)

            # train and forcast
            ifm=Informer_predict(data_pred[add_select_region],ifm_kwargs['context_length'],ifm_kwargs['prediction_length'])
            ifm.gen_src_tgt(freq,config,ifm_kwargs['train_dataloader_batch_size'],ifm_kwargs['test_dataloader_batch_size'])
            forecast_array,loss_history=ifm.predict(model,epochs=ifm_kwargs['epochs'],logging_steps=ifm_kwargs['logging_steps'])
            metric=ifm.evaluate(forecast_array)
            
            # 可视化各个区域的结果
            tab_list=st.tabs(add_select_region)
            for idx,tab in enumerate(tab_list):
                with tab:
                    st.subheader(f'{add_select_region[idx]}的预测结果')
                    ifm.plot(freq,idx,forecast_array)
            printMetric(model_name,metric)