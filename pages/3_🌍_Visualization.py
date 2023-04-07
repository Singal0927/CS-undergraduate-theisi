import streamlit as st
from pylab import mpl,plt
import plotly
from plot_class import Visualize
from streamlit_echarts import st_pyecharts
import time

# # 全球数据
# name=st.session_state['data']['Country/Region'];name.replace(name_map_world,inplace=True)
# value_df=st.session_state['data'].iloc[:,4:]
# world=Visualize(name=name,value_df=value_df,type='全球',max=1e4,location=[['Lat','Long']].to_numpy().tolist())
# tl=world('4/30/22')
# tl.render('world figure.html')

# 中国数据
data=st.session_state['data']

#绘图
st.line_chart(data)
history=st.checkbox('开始回顾！')

if history:
    vis=Visualize(name=data.columns,value_df=data,type='中国',max=1e4)
    vis_plot=vis('2022-4-30',dynamic=True)
    st_pyecharts(vis_plot)