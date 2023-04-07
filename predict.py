import pandas as pd
import numpy as np

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

import streamlit as st
from pylab import mpl,plt
plt.style.use('seaborn')

class ML_predict:
    def __init__(self,series:pd.Series,model,train_win,tgt_win,test_size=0.3) -> None:
        self.series=series
        self.model=model
        self.train_win=train_win
        self.tgt_win=tgt_win
        self.test_size=test_size
    
    def gen_seq_tgt(self):
        '''生成滞后项，np.ndarray'''
        seq,tgt=[],[]
        L=len(self.series)
        for i in range(L-self.train_win-self.tgt_win):
            seq.append(self.series.iloc[i:i+self.train_win].values)
            tgt.append(self.series.iloc[i+self.train_win:i+self.train_win+self.tgt_win].values)

        # 用于后续绘图
        self.time_index=self.series.index[:L-self.train_win-self.tgt_win]
        self.seq=np.array(seq)
        self.tgt=np.array(tgt)
    
    def predict(self):
        self.model_name=str(self.model).split('(')[0]
        xtrain,xtest,ytrain,ytest=train_test_split(self.seq,self.tgt,test_size=self.test_size,shuffle=False)

        self.model.fit(xtrain,ytrain)
        pred=self.model.predict(xtest)
        
        mse,mae,mape=mean_squared_error(ytest,pred),mean_absolute_error(ytest,pred),mean_absolute_percentage_error(ytest,pred)
        score={'MSE':round(mse,3),'MAE':round(mae,3),'MAPE':round(mape,3)}

        return pred,score
    
    def plot(self,pred,score):

        length_test=pred.shape[0]
        pred_idx=self.time_index[-length_test:]
        fig,ax=plt.subplots()
        ax.plot(self.time_index,self.seq[:,0],label='true values')
        ax.plot(pred_idx,pred,label='predictions')
        ax.vlines(pred_idx[0],ymin=self.seq[:,0].min(),ymax=self.seq[:,0].max(),colors='k',linestyles='dashed',alpha=0.3)
        ax.vlines(pred_idx[-1],ymin=self.seq[:,0].min(),ymax=self.seq[:,0].max(),colors='k',linestyles='dashed',alpha=0.3)
        # plt.text(pred_idx[0],pred[0],s=f'{pred_idx[0]}')
        ax.legend(fontsize=16)
        ax.set_title(f"{self.model_name}",fontsize=20)
        # plt.title(f"{self.model_name} MSE:{score['MSE']:.3f} MAE:{score['MAE']:.3f} MAPE:{score['MAPE']:.3f}",fontsize=20)
        st.pyplot(fig)
    
    def __call__(self):
        self.gen_seq_tgt()
        pred,score=self.predict()
        self.plot(pred,score)
        return score
    
class DL_predict(ML_predict):
    def __init__(self, series: pd.Series, model, train_win, tgt_win, test_size=0.3) -> None:
        super().__init__(series, model, train_win, tgt_win, test_size)