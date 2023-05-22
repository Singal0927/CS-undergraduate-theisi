import pandas as pd
import numpy as np
from tqdm import tqdm

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
from torch.utils.data import TensorDataset,DataLoader
from torch.optim import AdamW
from transformers import TimeSeriesTransformerConfig,TimeSeriesTransformerForPrediction
from transformers import PretrainedConfig

# 时间序列相关库
from gluonts.dataset.common import ListDataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
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


import streamlit as st
from pylab import mpl,plt
plt.style.use('seaborn')

class ML_predict:
    def __init__(self,df:pd.DataFrame,train_win,tgt_win) -> None:
        self.df=df
        self.train_win=train_win
        self.tgt_win=tgt_win

    def gen_src_tgt(self):
        '''生成滞后项，np.ndarray'''
        src,tgt=[],[]
        L=self.df.shape[0]
        for i in range(L-self.train_win-self.tgt_win):
            src.append(self.df.iloc[i:i+self.train_win].values)
            tgt.append(self.df.iloc[i+self.train_win:i+self.train_win+self.tgt_win].values)

        # 用于后续绘图
        if np.array(src).shape[-1]!=1:
            st.error('本网页中线性回归只适用于单变量预测！请删减变量！')
        else:
            self.src=np.array(src).squeeze(-1)
            self.tgt=np.array(tgt).squeeze(-1)
    
    def predict(self,model,test_size,batch_size=None):
        self.model_name=str(model).split('(')[0]
        xtrain,xtest,ytrain,ytest=train_test_split(self.src,self.tgt,test_size=test_size,shuffle=False)
        model.fit(xtrain,ytrain)
        pred=model.predict(xtest)
        
        mse,mae,mape=mean_squared_error(ytest,pred),mean_absolute_error(ytest,pred),mean_absolute_percentage_error(ytest,pred)
        score={'MSE':round(mse,3),'MAE':round(mae,3),'MAPE':round(mape,3)}
        return pred,score
    
    def plot(self,pred):

        fig=plt.figure(figsize=(10,5))
        plt.plot(self.df,label='true values')

        length_test=pred.shape[0]
        pred_idx=self.df.index[-length_test:]
        plt.plot(pred_idx,pred,label='predictions')

        plt.vlines(pred_idx[0],ymin=self.src[:,0].min(),ymax=self.src[:,0].max(),colors='k',linestyles='dashed',alpha=0.3)
        plt.vlines(pred_idx[-1],ymin=self.src[:,0].min(),ymax=self.src[:,0].max(),colors='k',linestyles='dashed',alpha=0.3)
        # plt.text(pred_idx[0],pred[0],s=f'{pred_idx[0]}')
        plt.legend(fontsize=16)
        plt.title(f"{self.model_name}",fontsize=20)
        # plt.title(f"{self.model_name} MSE:{score['MSE']:.3f} MAE:{score['MAE']:.3f} MAPE:{score['MAPE']:.3f}",fontsize=20)
        st.pyplot(fig)

    def __call__(self,model,test_size,batch_size=None):
        self.gen_src_tgt()
        pred,score=self.predict(model,test_size,batch_size)
        self.plot(pred)
        return score


class DL_predict(ML_predict):
    def __init__(self,df: pd.DataFrame,train_win, tgt_win) -> None:
        super().__init__(df,train_win, tgt_win)

    def gen_src_tgt(self):
        '''生成滞后项，np.ndarray'''
        src,tgt,tgt_y=[],[],[]
        L=self.df.shape[0]
        for i in range(L-self.train_win-self.tgt_win):
            src.append(self.df.iloc[i : i + self.train_win].values)
            tgt.append(self.df.iloc[i + self.train_win - 1 : i + self.train_win + self.tgt_win -1].values)
            tgt_y.append(self.df.iloc[i + self.train_win : i + self.train_win + self.tgt_win].values)

        self.src=torch.Tensor(np.array(src))
        self.tgt=torch.Tensor(np.array(tgt))
        self.tgt_y=torch.Tensor(np.array(tgt_y))

    def predict(self, model,test_size,batch_size):
        self.model_name=str(model).split('(')[0]

        # 划分数据集并构造dataset与dataloader
        xtrain,xtest,tgttrain,tgttest,ytrain,ytest=train_test_split(self.src,self.tgt,self.tgt_y,test_size=test_size,shuffle=False)
        train_ds=TensorDataset(torch.Tensor(xtrain),torch.Tensor(tgttrain),torch.Tensor(ytrain))
        train_dl=DataLoader(train_ds,batch_size=batch_size)

        # 训练
        criterion=nn.MSELoss()
        optimizer=torch.optim.Adam(model.parameters(),lr=1)
        tgt_mask=self.gen_square_subsequent_mask(self.tgt_win,self.tgt_win)
        memory_mask=self.gen_square_subsequent_mask(self.tgt_win,self.train_win)
        model.train()
        for epoch in range(1):
            print(f"epoch: {epoch+1}")
            print('-'*100)
            pbar=tqdm(train_dl)
            for src,tgt,tgt_y in pbar:
                tgt_pred=model(src,tgt,tgt_mask=tgt_mask,memory_mask=memory_mask)
                loss=criterion(tgt_pred,tgt_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix_str(f'当前batch的loss为：{loss.item():.3f}')

        # 推理并计算metric
        model.eval()
        with torch.no_grad():
            pred=model(xtest,tgttest,tgt_mask=tgt_mask,memory_mask=memory_mask).detach().numpy()

            # 多变量时间预测
            mse_lis,mae_lis,mape_lis=[],[],[]
            for nvar in range(pred.shape[-1]):
                mse_lis.append(mean_squared_error(ytest[:,:,nvar],pred[:,:,nvar]))
                mae_lis.append(mean_absolute_error(ytest[:,:,nvar],pred[:,:,nvar]))
                mape_lis.append(mean_absolute_percentage_error(ytest[:,:,nvar],pred[:,:,nvar]))

            score={'MSE':round(np.mean(mse_lis),3),'MAE':round(np.mean(mae_lis),3),'MAPE':round(np.mean(mape_lis),3)}
            return pred.squeeze(1),score
        
    def gen_square_subsequent_mask(self,dim1,dim2):
        '''
        Args:
        dim1: int, for both src and tgt masking, this must be target sequence
              length
        dim2: int, for src masking this must be encoder sequence length (i.e. 
              the length of the input sequence to the model), 
              and for tgt masking, this must be target sequence length 
        Return:
            A Tensor of shape [dim1, dim2]
        '''
        return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)
        # return torch.triu(torch.ones(dim1,dim2),diagonal=1).to(torch.bool)#这种计算mask的方式不行


class TST_predict(ML_predict):
    def __init__(self, series: pd.Series, train_win, tgt_win,) -> None:
        super().__init__(series, train_win, tgt_win)
        self.series=series
        self.context_length=train_win
        self.prediction_length=tgt_win

    def gen_src_tgt(self,freq,config,train_dataloader_batch_size=64,test_dataloader_batch_size=16):
        '''在🤗框架下生成dataset，并经transformation后返回dataloader'''
        target=self.series.to_numpy().T # shape=[num_series,num_steps],Convention: time axis is always the last axis.
        start_series=self.series.index

        # define the dataset
        self.train_ds=ListDataset(
            [
                {   
                    FieldName.START:start_series[0],
                    FieldName.TARGET:target
                }
                for target in target[:,:-self.prediction_length]
            ],
            freq=freq
        )
        self.test_ds=ListDataset(
            [
                {   
                    FieldName.START:start_series[0],
                    FieldName.TARGET:target
                }
                for target in target
            ],
            freq=freq
        )
        
        # group a univariate dataset into a single multivariate time series: [n_vars,time_steps]
        n_vars=len(self.train_ds)
        train_grouper=MultivariateGrouper(max_target_dim=n_vars)
        test_grouper=MultivariateGrouper(max_target_dim=n_vars)
        self.multivariate_train_ds=train_grouper(self.train_ds)
        self.multivariate_test_ds=test_grouper(self.test_ds)
        
        # define the dataloader 
        self.train_dataloader=createTrainDataLoader(config,freq,self.multivariate_train_ds,batch_size=16,num_batches_per_epoch=100)
        self.test_dataloader=createTestDataLoader(config,freq,self.multivariate_test_ds,batch_size=4)

    def predict(self, model, epochs=5,logging_steps=10):
        # train
        device='cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        optimizer=AdamW(model.parameters(),lr=6e-4,betas=(0.9,0.95),weight_decay=1e-1)

        total_steps=0
        loss_history=[]
        model.train()
        for epoch in range(epochs):
            for batch in self.train_dataloader:
                if batch['past_values'].shape[-1]==1:#只有一个变量
                    batch['past_values']=batch['past_values'].squeeze(2)
                    batch['past_observed_mask']=batch['past_observed_mask'].squeeze(2)
                    batch['future_values']=batch['future_values'].squeeze(2)
                    batch['future_observed_mask']=batch['future_observed_mask'].squeeze(2)

                output=model(
                    past_values=batch['past_values'].to(device),
                    past_time_features=batch['past_time_features'].to(device),
                    past_observed_mask=batch['past_observed_mask'].to(device),
                    future_time_features=batch['future_time_features'].to(device),
                    future_values=batch['future_values'].to(device),
                    future_observed_mask=batch['future_observed_mask'].to(device),
                )

                loss=output.loss
                loss_history.append(loss.item())
                loss.backward()
                optimizer.step()

                total_steps+=1
                if total_steps%logging_steps==0:
                    st.write(loss.item())

        # inference
        model.eval()
        forecast_lis=[]# (batch_size,1,number of sample paths,prediction length)
        for batch in self.test_dataloader:
            if batch['past_values'].shape[-1]==1:#只有一个变量
                batch['past_values']=batch['past_values'].squeeze(2)
                batch['past_observed_mask']=batch['past_observed_mask'].squeeze(2)

            output=model.generate(
                past_values=batch['past_values'].to(device),
                past_time_features=batch['past_time_features'].to(device),
                past_observed_mask=batch['past_observed_mask'].to(device),
                future_time_features=batch['future_time_features'].to(device),
            ).sequences
            if len(output.shape)==3:#表明是变量个数是1
                output=output.unsqueeze(-1)

            forecast_lis.append(output.cpu().detach().numpy())
        forecast_array=np.vstack(forecast_lis)# after np.vstack, forecast_array'shape becomes (batch_size,number of sample paths,prediction length,n_vars))
        return forecast_array,loss_history

    def evaluate(self,forecast_array):
        mse_lis,mae_lis,mape_lis=[],[],[]
        for idx,forecast in enumerate(forecast_array):#forecast: (number of sample paths,prediction length,n_vars))
            forecast_mean=np.mean(forecast,axis=0).T#forecast_mean: (n_vars, prediction length)
            st.write('forecast_mean',forecast_mean.shape)
            
            gold=self.multivariate_test_ds[idx]['target'][:,-self.prediction_length:]
            st.write('gold',gold.shape)
            
            mse_lis.append(mean_squared_error(gold,forecast_mean))
            mae_lis.append(mean_absolute_error(gold,forecast_mean))
            mape_lis.append(mean_absolute_percentage_error(gold,forecast_mean))

        score={'MSE':round(np.mean(mse_lis),3),'MAE':round(np.mean(mae_lis),3),'MAPE':round(np.mean(mape_lis),3)}
        return score
    
    def plot(self, freq,mv_index,forecast_array):
        fig,axes=plt.subplots()
        index=pd.period_range(
            start=self.multivariate_test_ds[0]['start'],
            periods=len(self.multivariate_test_ds[0]['target'][mv_index]),
            freq=freq,
        ).to_timestamp()

        # true values
        axes.plot(index[-self.context_length:],self.multivariate_test_ds[0]['target'][mv_index,-self.context_length:],label='true values')

        # preditions
        forecast_mean=np.mean(forecast_array[0],axis=0)[:,mv_index]
        axes.plot(index[-self.prediction_length:],forecast_mean[-self.prediction_length:],label='preditions')

        # shadow area
        axes.fill_between(
            index[-self.prediction_length:],
            y1=forecast_mean-forecast_mean.std(),
            y2=forecast_mean+forecast_mean.std(),
        )

        plt.legend()
        st.pyplot(fig)

class Informer_predict(TST_predict):
    def __init__(self, series: pd.Series, train_win, tgt_win) -> None:
        super().__init__(series, train_win, tgt_win)

def createTransformation(config:PretrainedConfig,freq):
    return Chain(
        [
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # temporal features serve as positional encodings
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(freq),# 返回len(time_features_from_frequency_str(freq))*len(train_ds)的array
                pred_length=config.prediction_length,
            ),
            # another temporal feature
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,#age feature grows logarithmically otherwise linearly overtime.
            ),
            # vertically stack all the temporal features into FieldName.FEAT_TIME
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME,FieldName.FEAT_AGE],
                h_stack=False,#dim=0 if h_stack=False else dim=1
            ),
            RenameFields(
                mapping={
                    FieldName.OBSERVED_VALUES:'observed_mask',
                    FieldName.FEAT_TIME:'time_features',
                    FieldName.TARGET:'values'
                }
            )
        ]
    )
    
def createInstanceSplitter(config:PretrainedConfig,mode,train_sampler=None,validation_sampler=None):
    assert mode in ['train','validation','test']

    instance_sampler={
        'train':train_sampler or ExpectedNumInstanceSampler(num_instances=1,min_future=config.prediction_length),
        'validation':validation_sampler or ValidationSplitSampler(min_future=config.prediction_length),
        'test':TestSplitSampler(),
    }[mode]

    return InstanceSplitter(
        target_field='values',
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config.context_length+max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=['time_features','observed_mask']
    )

def createTrainDataLoader(config:PretrainedConfig,freq,data,batch_size,num_batches_per_epoch,shuffle_buffer_length=None):
    prediction_input_names=[
        'past_values',
        'past_time_features',
        'past_observed_mask',
        'future_time_features',
    ]
    if config.num_static_categorical_features>0:
        prediction_input_names.append('static_categorical_features')
    
    if config.num_static_real_features>0:
        prediction_input_names.append('static_real_features')

    training_input_names=prediction_input_names+[
        'future_values',
        'future_observed_mask',
    ]
    
    transformation=createTransformation(freq=freq,config=config)
    transformed_data=transformation.apply(data,is_train=True)
    
    instance_splitter=createInstanceSplitter(config,'train')+SelectFields(training_input_names)#在所选择的fields上create InstanceSplitter

    train_instances=instance_splitter.apply(
        Cyclic(transformed_data) if shuffle_buffer_length is None
        else PseudoShuffled(Cyclic(transformed_data),shuffle_buffer_length=shuffle_buffer_length)
    )

    return IterableSlice(
        iter(DataLoader(
            IterableDataset(train_instances),
            batch_size=batch_size,
        )),num_batches_per_epoch,#表示一次取num_batches_per_epoch个batches
    )

def createTestDataLoader(config,freq,data,batch_size):
    prediction_input_names=[
        'past_values',
        'past_time_features',
        'past_observed_mask',
        'future_time_features',
    ]
    if config.num_static_categorical_features>0:
        prediction_input_names.append('static_categorical_features')
    
    if config.num_static_real_features>0:
        prediction_input_names.append('static_real_features')
    
    transformation=createTransformation(freq=freq,config=config)
    transformed_data=transformation.apply(data,is_train=False)
    
    instance_splitter=createInstanceSplitter(config,'test')+SelectFields(prediction_input_names)#在所选择的fields上create InstanceSplitter

    test_instances=instance_splitter.apply(transformed_data,is_train=False)

    return DataLoader(IterableDataset(test_instances),batch_size=batch_size)