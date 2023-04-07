import pandas as pd
from typing import List,Union,Optional

from pyecharts.charts import Bar,Map,Timeline,Pie,Geo,MapGlobe,Map3D,Page,Grid
import pyecharts.options as opts
from pyecharts.faker import Faker
import streamlit as st

#demo为累计确诊
class Visualize:
    def __init__(self,name:List[str],value_df:pd.DataFrame,type:str='全球',max:int=1e3):
        '''
        name：城市或国家（区域）名称
        value：列名为日期
        type：为图表类型，从“中国”与“全球”中择其一
        '''
        self.name=name
        self.value_df=value_df
        self.type=type
        if self.type=='全球':
            self.region='world'
        else:
            self.region='china'
        # self.max=self.value_df.max().max()
        self.max=max
        self.min=self.value_df.min().min()
        self.feature='累计确诊'

    def map(self,date):
        value=self.value_df.loc[date]
        name=list(value.index)
        value=list(value)
        map=(
            Map(init_opts=opts.InitOpts(width='700px',height='400px'))
            .add(f'{date}{self.feature}',[list(z) for z in zip(name,value)],self.region,is_map_symbol_show=False)
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))#不展示国家或地区名称
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"{self.type}疫情地图"),
                visualmap_opts=opts.VisualMapOpts(max_=self.max),
            )
        )
        return map

    def bar(self,date):
        value=self.value_df.loc[date].sort_values(ascending=False)[:10]
        name=list(value.index)
        value=list(value)
        bar=(
            Bar(init_opts=opts.InitOpts(width='350px',height='400px'))
            .add_xaxis(list(name))
            .add_yaxis(
                series_name=f'{date}{self.feature}',
                y_axis=list(value),
                is_selected=True,#选中图例
                label_opts=opts.LabelOpts(is_show=False),
                # is_realtime_sort=True,
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-90)),
                title_opts=opts.TitleOpts(title=f'{self.type}疫情柱状图', subtitle="数据来自约翰霍普金斯大学"),
                tooltip_opts=opts.TooltipOpts(is_show=True, trigger="axis", axis_pointer_type="shadow"),
            )
        )
        return bar
    
    def pie(self,date):
        value=self.value_df.loc[date].sort_values(ascending=False)[:10]
        name=list(value.index)
        value=list(value)
        pie=(
            Pie(init_opts=opts.InitOpts(width='350px',height='400px'))
            .add(
                series_name=f'{date}{self.feature}饼状图',
                data_pair=[list(z) for z in zip(name,value)],
                center=["90%", "25%"],
                radius="35%",
            )
        )
        return pie
    
    def map3D(self,date):
        '''不支持TimeLine，请输入具体日期'''
        pass
        # for date in self.value_df.columns:
        #     value=list(self.value_df[date])
        #     map3d = (
        #         Map3D()
        #         .add_schema(
        #             maptype=self.region,
        #             itemstyle_opts=opts.ItemStyleOpts(
        #                 color="rgb(5,101,123)",
        #                 opacity=1,
        #                 border_width=0.8,
        #                 border_color="rgb(62,215,213)",
        #             ),
        #             map3d_label=opts.Map3DLabelOpts(
        #                 is_show=False,
        #                 formatter=JsCode("function(data){return data.name + " " + data.value[2];}"),
        #             ),
        #             emphasis_label_opts=opts.LabelOpts(
        #                 is_show=False,
        #                 color="#fff",
        #                 font_size=10,
        #                 background_color="rgba(0,23,11,0)",
        #             ),
        #             light_opts=opts.Map3DLightOpts(
        #                 main_color="#fff",
        #                 main_intensity=1.2,
        #                 main_shadow_quality="high",
        #                 is_main_shadow=False,
        #                 main_beta=10,
        #                 ambient_intensity=0.3,
        #             ),
        #         )
        #         .add(
        #             maptype=self.region,
        #             series_name=f'{self.type}{self.feature}三维分布图',
        #             data_pair=[[z[0],[z[1][1],z[1][0],z[2]]] for z in zip(self.name,self.location,value)],
        #             type_=ChartType.BAR3D,
        #             bar_size=1,
        #             shading="lambert",
        #             is_animation=True,
        #             label_opts=opts.LabelOpts(
        #                 is_show=False,
        #                 formatter=JsCode("function(data){return data.name + ' ' + data.value[2];}"),
        #             ),
        #         )
        #         .set_global_opts(title_opts=opts.TitleOpts(title="Map3D-Bar3D"))
        #     )
        # return map3d

    def mapGlobe(self,date):
        '''只支持全球地图，请输入具体日期'''
        # mapGlobe不能和其他的图一起放在同一个网页内。。。。
        if self.type=='全球':
            
            tl=Timeline()
            # for date in self.value_df.columns:
            value=list(self.value_df.loc[date])
            map_globe=(
                MapGlobe()
                .add_schema()
                .add(
                    maptype=self.region,
                    series_name=f'{self.type}{self.feature}全球分布图',
                    data_pair=[list(z) for z in zip(list(self.name),value)],
                    is_map_symbol_show=False,#展示每个国家上的圆圈
                    label_opts=opts.LabelOpts(is_show=False)#不展示国家或地区名称
                )
                .set_global_opts(
                    visualmap_opts=opts.VisualMapOpts(
                        min_=self.min,
                        max_=self.max,
                        range_text=["max", "min"],
                        is_calculable=True,
                        range_color=["lightskyblue", "yellow", "orangered"],
                    )
                )
            )
            return map_globe
        else:
            return None

    def __call__(self,date,dynamic=False):
        page=Page(layout=Page.DraggablePageLayout, page_title="疫情数据动态展示")
        if dynamic:
            tl1=Timeline();tl2=Timeline()

            for date in self.value_df.index:
                # map=self.map(date)
                # tl1.add(map,f'{date}{self.feature}平面图')
                # tl1.add_schema(is_auto_play=True, play_interval=10,is_loop_play=False)

                bar=self.bar(date)
                pie=self.pie(date)
                tl2.add(bar.overlap(pie),f'{date}{self.feature}柱形图与饼状图')
                tl2.add_schema(is_auto_play=True, play_interval=10,is_loop_play=False)

            
            grid=(
                Grid(init_opts=opts.InitOpts(width="1200px", height="800px"))
                # .add(map_globe,grid_opts=opts.GridOpts(pos_right="58%"), is_control_axis_index=True)
                # .add(map,grid_opts=opts.GridOpts(pos_top="50%", pos_bottom="100%"), is_control_axis_index=True)
                .add(tl2,grid_opts=opts.GridOpts(pos_top="20%", pos_bottom="0%"), is_control_axis_index=True)
            )
            return grid
        else:
            # map_globe=self.mapGlobe(date)
            map=self.map(date)
            bar=self.bar(date)
            pie=self.pie(date)
            grid=(
                Grid(init_opts=opts.InitOpts(width="1200px", height="800px"))
                # .add(map_globe,grid_opts=opts.GridOpts(pos_right="58%"), is_control_axis_index=True)
                
                .add(bar.overlap(pie),grid_opts=opts.GridOpts(pos_top="0%", pos_bottom="50%"), is_control_axis_index=True)
                # .add(map,grid_opts=opts.GridOpts(pos_top="50%", pos_bottom="100%"))
            )
            return grid


if __name__=='__main__':
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


    # 全球数据
    world=Visualize(name=raw.columns,value_df=raw,type='中国',max=1e4)
    tl=world('2022/4/30')
    tl.render('world figure.html')

    # # 中国数据
    # data=data.loc[data['Country/Region']=='China']
    # data.dropna(inplace=True)
    # name=data['Province/State'];name.replace(name_map_cn,inplace=True)
    # value_df=data.iloc[:,4:]
    # world=Visualize(name=name,value_df=value_df,type='中国',max=1e4,location=data[['Lat','Long']].to_numpy().tolist())
    # tl=world.bar('4/30/22')
    # tl.render('china figure.html')