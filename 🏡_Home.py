# 运行streamlit hello 以查看案例
# module在多个页面间全局共享，但set_page_config不行
import pandas as pd
import numpy as np
import streamlit as st
import streamlit_echarts as se
from datetime import time,datetime

raw=pd.read_csv('demo.csv')
raw.index=pd.DatetimeIndex(raw['Unnamed: 0'])
del raw['Unnamed: 0']
raw.index.rename('date',inplace=True)
st.set_page_config(page_title="毕业论文——葛新杰",layout="centered")

#
name_map_world={'US':'United States'}
name_map_cn={'Heilongjiang': '黑龙江省', 'Jilin': '吉林省', 'Liaoning': '辽宁省', 'Beijing': '北京市', 
            'Tianjin': '天津市', 'Hebei': '河北省', 'Shanxi': '山西省', 'Inner Mongolia': '内蒙古自治区', 
            'Shanghai': '上海市', 'Jiangsu': '江苏省', 'Shandong': '山东省', 'Zhejiang': '浙江省', 
            'Anhui': '安徽省', 'Jiangxi': '江西省', 'Fujian': '福建省', 'Guangdong': '广东省', 
            'Macau': '澳门特别行政区', 'Taiwan': '台湾省', 'Hong Kong': '香港特别行政区', 'Tibet': '西藏自治区', 
            'Guangxi': '广西省', 'Hainan': '海南省', 'Henan': '河南省', 'Hubei': '湖北省', 'Hunan': '湖南省', 
            'Shaanxi': '陕西省', 'Xinjiang': '新疆自治区', 'Ningxia': '宁夏自治区', 'Gansu': '甘肃省', 
            'Qinghai': '青海省', 'Chongqing': '重庆市', 'Sichuan': '四川省', 'Guizhou': '贵州省', 'Yunnan': '云南省',}
raw.rename(columns=name_map_cn,inplace=True)
del raw['Unknown']

#设置侧边栏
st.sidebar.markdown('欢迎使用疫情数据分析工具！')

#设置标题
st.header('计科毕业论文')
st.subheader('葛新杰I01914212')

#公共数据
if 'data' not in st.session_state:
    st.session_state['data']=None

with st.form('select data range'):
    time_range=st.slider(
        '请选择起止时间：',
        value=(datetime(2020,6,1),datetime(2023,3,1)),
        format='YYYY/MM/DD',
    )

    country_range=st.multiselect(
        '请选择省份：',
        raw.columns.to_list(),
        ['安徽省','广东省','北京市'],
    )
    submitted=st.form_submit_button('提交')
    if submitted:
        st.write('起止时间：',time_range)
        st.write('所选省份：',country_range)
        data=raw.loc[time_range[0]:time_range[1],country_range]
        
        st.session_state['data']=data
        st.dataframe(st.session_state['data'].sample(5))
    else:
        st.write('您还未选择数据！')


# country='Afghanistan'
# data=raw[raw['Country/Region']==country].iloc[:,4:].T

# st.write(pd.DatetimeIndex(data.index))
# yes=st.checkbox('进行以下步骤')