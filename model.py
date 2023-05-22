# !/usr/bin/env python3
"""
==== No Bugs in code, just some Random Unexpected FEATURES ====
┌─────────────────────────────────────────────────────────────┐
│┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
│├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
│├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
│├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
│└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
│      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
│      └───┴─────┴───────────────────────┴─────┴───┘          │
└─────────────────────────────────────────────────────────────┘

UIE torch版本实现，包含模型预处理/后处理函数。

Author: pankeyu
Date: 2022/10/18
"""
import json
from typing import List

import torch
import torch.nn as nn
import numpy as np


class UIE(nn.Module):

    def __init__(self, encoder):
        """
        init func.

        Args:
            encoder (transformers.AutoModel): backbone, 默认使用 ernie 3.0
        
        Reference:
            https://github.com/PaddlePaddle/PaddleNLP/blob/a12481fc3039fb45ea2dfac3ea43365a07fc4921/model_zoo/uie/model.py
        """
        super().__init__()
        self.encoder = encoder
        hidden_size = 768
        self.linear_start = nn.Linear(hidden_size, 1)
        self.linear_end = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        input_ids: torch.tensor,
        token_type_ids: torch.tensor,
        attention_mask=None,
        pos_ids=None,
    ) -> tuple:
        """
        forward 函数，返回开始/结束概率向量。

        Args:
            input_ids (torch.tensor): (batch, seq_len)
            token_type_ids (torch.tensor): (batch, seq_len)
            attention_mask (torch.tensor): (batch, seq_len)
            pos_ids (torch.tensor): (batch, seq_len)

        Returns:
            tuple:  start_prob -> (batch, seq_len)
                    end_prob -> (batch, seq_len)
        """
        sequence_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=pos_ids,
            attention_mask=attention_mask,
        )["last_hidden_state"]
        start_logits = self.linear_start(sequence_output)       # (batch, seq_len, 1)
        start_logits = torch.squeeze(start_logits, -1)          # (batch, seq_len)
        start_prob = self.sigmoid(start_logits)                 # (batch, seq_len)
        end_logits = self.linear_end(sequence_output)           # (batch, seq_len, 1)
        end_logits = torch.squeeze(end_logits, -1)              # (batch, seq_len)
        end_prob = self.sigmoid(end_logits)                     # (batch, seq_len)
        return start_prob, end_prob
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model,max_seq_len=5000,dropout=0.1,batch_first=False) -> None:
        '''不知道为什么，d_model必须是偶数'''
        super().__init__()
        self.x_dim=1 if batch_first else 0
        self.batch_first=batch_first
        pos=torch.arange(0,max_seq_len).unsqueeze(1)
        div_term=torch.pow(10000,torch.arange(0,d_model,2)/d_model)
        
        if self.batch_first:
            pe=torch.zeros(1,max_seq_len,d_model)
            pe[0,:,0::2]=torch.sin(pos/div_term)
            pe[0,:,1::2]=torch.cos(pos/div_term)
        else:
            pe=torch.zeros(max_seq_len,1,d_model)
            pe[:,0,0::2]=torch.sin(pos/div_term)
            pe[:,0,1::2]=torch.cos(pos/div_term)
        self.register_buffer('pe', pe)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        if self.batch_first:
            x=x+self.pe[:,:x.shape[self.x_dim]]
        else:
            x=x+self.pe[:x.shape[self.x_dim]]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    def __init__(self,nvars,d_model,d_hid,nheads,nlayers,dropout=0.1) -> None:
        '''把d_modal与变量个数做映射
        '''
        super().__init__()
        self.model_name='Transformer Encoder'
        # self.max_seq_len=train_win

        # encoder
        self.encoder_input_mapping=nn.Linear(nvars,d_model)
        self.pos_encoding=PositionalEncoding(d_model=d_model,batch_first=True)
        encoder_layer=nn.TransformerEncoderLayer(d_model,nheads,d_hid,dropout,batch_first=True)
        self.transformer_encoder=nn.TransformerEncoder(encoder_layer,num_layers=nlayers,norm=None)
        #norm参数表示每个encoder_layer所需的normalization方法，但是nn.TransformerEncoderLayer已经包含了Layer-normalization，故不需要向norm传递参数，事实上，norm参数是为不具备normalization方法的custorm encoder-layers准备的。
        
        # decoder
        self.decoder_input_mapping=nn.Linear(nvars,d_model)
        decoder_layer=nn.TransformerDecoderLayer(d_model=d_model,nhead=nheads,dim_feedforward=d_hid,dropout=dropout,batch_first=True)
        self.transformer_decoder=nn.TransformerDecoder(decoder_layer,num_layers=nlayers,norm=None)
        self.decoder_output_mapping=nn.Linear(d_model,nvars)

        # 初始化参数
        self.init_weight()
    
    def init_weight(self):
        init_range=0.1
        self.encoder_input_mapping.weight.data.uniform_(-init_range,init_range)
        self.encoder_input_mapping.bias.data.zero_()

        self.decoder_input_mapping.weight.data.uniform_(-init_range,init_range)
        self.decoder_input_mapping.bias.data.zero_()

        self.decoder_output_mapping.weight.data.uniform_(-init_range,init_range)
        self.decoder_output_mapping.bias.data.zero_()

    def forward(self,src,tgt,memory_mask=None,tgt_mask=None):
        '''
        Return a tensor of shape:
        [batch_size,tgt_win,nvars]
        Args:
            src:[batch_size,train_win,nvars]
            tgt:[batch_size,tgt_win,nvars]
            memory_mask:[train_win,train_win]
            tgt_mask:[tgt_win,tgt_win]
        '''
        # encoder
        src=self.encoder_input_mapping(src)
        src=self.pos_encoding(src)
        encoder_output=self.transformer_encoder(src)

        # decoder
        tgt=self.decoder_input_mapping(tgt)
        decoder_output=self.transformer_decoder(tgt=tgt,memory=encoder_output,tgt_mask=tgt_mask,memory_mask=memory_mask)#memory：最后一层encoder layer的输出
        decoder_output=self.decoder_output_mapping(decoder_output)
        return decoder_output