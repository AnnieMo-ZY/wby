import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings
from tensorflow import keras
warnings.filterwarnings('ignore')
from datetime import date
import pandas_ta as ta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import time
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import os
# new
from pyecharts.charts import *
from pyecharts import options as opts
from streamlit_echarts import st_pyecharts
import functions as F

# pyechart tutorial
#https://www.heywhale.com/mw/project/5eb7958f366f4d002d783d4a
st.set_page_config(page_title = '📈 AI Guided Trading System',layout = 'wide')

    
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown('# 📈AI Guided Financial Trading Dashboard')

st.markdown('#### 通过检测Long Term Moving Average(长期移动均线)与Short Term Moving Average(短期移动均线)交叉的交易情况,')
st.markdown('#### 结合RSI,MACD,WR等11类技术指标对交易市场,使用RNN(循环神经网络),Transfomer(注意力机制)模型,对下一时刻的最高价/最低价进行预测,以及预测进场时机')
st.markdown('***原模型训练集为2018~2020年外汇市场M15货币数据***')

# handle data input / select perfer stock 
stock_name = st.text_input('输入股票代号: ' , help = '查阅股票代号: https://finance.yahoo.com/lookup/')
STOCK = yf.Ticker('XPEV')

if stock_name:
    STOCK = yf.Ticker(stock_name)

data = STOCK.history(interval = "15m")
data['Datetime'] = data.index
data['Datetime'] = data['Datetime'].astype(str)

# convert to Asia timezone
# data['Datetime'] = pd.DataFrame(pd.to_datetime(data['Datetime'] ,utc=True).tz_convert('Asia/Shanghai')).index
data = F.pre_process(data)
g = (Kline(init_opts=opts.InitOpts(width="900px", height='500px'))
        .add_xaxis(data['Datetime'].tolist()) 
        #y轴数据，默认open、close、low、high，转为list格式
        .add_yaxis("",y_axis=data[['Open', 'Close', 'Low', 'High']].values.tolist(),
        itemstyle_opts=opts.ItemStyleOpts(
        color="rgb(205,51,0)",#阳线红色 涨 #FF0000
        color0="rgb(69,139,116)",#阴线绿色 跌 #32CD32
        border_color="rgb(205,51,0)",
        border_color0="rgb(69,139,116)",),)
        .set_global_opts(
        #标题
        title_opts =opts.TitleOpts(title = f'{stock_name} K线图',
        subtitle = '15M',pos_left = 'left',title_textstyle_opts = opts.TextStyleOpts(font_size=28)),
        # 图例
        legend_opts=opts.LegendOpts(
            is_show=False, pos_bottom=10, pos_left="center"),
        # 缩放
        datazoom_opts=[
            opts.DataZoomOpts(
                is_show=False,
                type_="inside",
                xaxis_index=[0, 1],
                range_start=98,
                range_end=100,
            ),
            opts.DataZoomOpts(
                is_show=True,
                xaxis_index=[0, 1],
                type_="slider",
                pos_top="85%",
                range_start=98,
                range_end=100,)
                ,],)       
    )


tab0, tab1, tab2, tab3= st.tabs(['数据','K线图', '技术指标','预测模型'])
with tab0:
    st.dataframe(data.iloc[::-1], height=600,use_container_width = True)

with tab1:
    st_pyecharts(g,width="100%", height='900px')
    st.markdown('#### 假如RSI处于超卖区域并开始上穿30水平,则你需要寻找看涨的反转烛台形态。此时为买入信号')
    st.markdown('#### 如果RSI处于超买区域并开始下穿70水平,则需要开始观察寻找看跌反转烛台。此时为做空信号')
    rsi_fig = F.RSI_plot(data)
    st.plotly_chart(rsi_fig,use_container_width = True)

with tab2:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(str(data['Datetime'].values[-2])[11:-6] + " 开盘价", str(data.Open.values[-2])[0:7], str(data.Open.values[-2] -data.Open.values[-3])[0:7])
    col2.metric(str(data['Datetime'].values[-2])[11:-6] + " 收盘价", str(data.Close.values[-2])[0:7],  str(data.Close.values[-2] -data.Close.values[-3])[0:7])
    col3.metric(str(data['Datetime'].values[-2])[11:-6] + " 最高价", str(data.High.values[-2])[0:7], str(data.High.values[-2] -data.High.values[-3])[0:7])
    col4.metric(str(data['Datetime'].values[-2])[11:-6] + " 最低价", str(data.Low.values[-2])[0:7], str(data.Low.values[-2] -data.Low.values[-3])[0:7])
    
    col1.metric(str(data['Datetime'].values[-2])[11:-6] + " RSI15", str(data.RSI_15.values[-2])[0:7], str(data.RSI_15.values[-2] - data.RSI_15.values[-3])[0:7])
    col2.metric(str(data['Datetime'].values[-2])[11:-6] + " WR15", str(data.wr15.values[-2])[0:7],  str(data.wr15.values[-2] - data.wr15.values[-3])[0:7])
    col3.metric(str(data['Datetime'].values[-2])[11:-6] + " ATR15", str(data.atr15.values[-2])[0:7], str(data.atr15.values[-2] -data.atr15.values[-3])[0:7])
    col4.metric(str(data['Datetime'].values[-2])[11:-6] + " SMA15", str(data.sma15.values[-2])[0:7], str(data.sma15.values[-2] -data.sma15.values[-3])[0:7])

    col1.metric(str(data['Datetime'].values[-2])[11:-6] + " RSI25", str(data.RSI_25.values[-2])[0:7], str(data.RSI_25.values[-2] - data.RSI_25.values[-3])[0:7])
    col2.metric(str(data['Datetime'].values[-2])[11:-6] + " WR25", str(data.wr25.values[-2])[0:7],  str(data.wr25.values[-2] - data.wr25.values[-3])[0:7])
    col3.metric(str(data['Datetime'].values[-2])[11:-6] + " ATR25", str(data.atr25.values[-2])[0:7], str(data.atr15.values[-2] -data.atr25.values[-3])[0:7])
    col4.metric(str(data['Datetime'].values[-2])[11:-6] + " SMA25", str(data.sma25.values[-2])[0:7], str(data.sma15.values[-2] -data.sma25.values[-3])[0:7])

with tab3:
    WINDOW_SIZE = 10
    
    st.markdown('### 模型特征: ')
    st.dataframe(data)

    LABEL_MODEL = st.button('RNN模型预测')

    # file path "//app//wby//RNN.h5"
    model = keras.models.load_model("C:\\Users\\HFY\\Desktop\\app\\Bi_RNN.h5", compile=False)

    if LABEL_MODEL :
        with st.spinner(text="##### 正在处理数据..."):
            data = F.pre_process(data)
            train_x_dict, price_scaler_max,price_scaler_min = F.generate_sequence(data,WINDOW_SIZE)
            predicted_max,predicted_min,predicted_label = F.make_prediction(model,train_x_dict,price_scaler_min,price_scaler_max)
            st.success('🚩已完成')
            
        # check model performance
        max_chart_data = pd.DataFrame({'预测最高值':[float(i) for i in predicted_max] , '真实最高值':data[f'max_{WINDOW_SIZE}'].tolist()[:len(data) - WINDOW_SIZE+1]})
        st.markdown('### 预测最高值验证:')
        st.line_chart(max_chart_data)

        min_chart_data = pd.DataFrame({'预测最低值':[float(i) for i in predicted_min] , '真实最低值':data[f'min_{WINDOW_SIZE}'].tolist()[:len(data) - WINDOW_SIZE+1]})
        st.markdown('### 预测最低值验证:')
        st.line_chart(min_chart_data)
        marker_ls = F.label_to_marker(data,predicted_label)
        h = (Kline(init_opts=opts.InitOpts(width="900px", height='500px'))
                .add_xaxis(data['Datetime'].tolist()) 
                #y轴数据，默认open、close、low、high，转为list格式
                .add_yaxis("",y_axis=data[['Open', 'Close', 'Low', 'High']].values.tolist(),
                itemstyle_opts=opts.ItemStyleOpts(
                color="rgb(205,51,0)",#阳线红色 涨 #FF0000
                color0="rgb(69,139,116)",#阴线绿色 跌 #32CD32
                border_color="rgb(205,51,0)",
                border_color0="rgb(69,139,116)",),)
                .set_series_opts(
                # 为了不影响标记点，这里把标签关掉
                label_opts=opts.LabelOpts(is_show=False,position = 'outside'),
                markpoint_opts=opts.MarkPointOpts(
                symbol = 'pin',
                symbol_size = [35,35],
                # 根据coord坐标对应轴内 xy对应像素坐标
                data=marker_ls
                )
                )
                .set_global_opts(
                #标题
                title_opts =opts.TitleOpts(title = f'{stock_name} K线图',
                subtitle = '15M',pos_left = 'left',title_textstyle_opts = opts.TextStyleOpts(font_size=28)),
                # 图例
                legend_opts=opts.LegendOpts(
                    is_show=False, pos_bottom=10, pos_left="center"),
                # 缩放
                datazoom_opts=[
                    opts.DataZoomOpts(
                        is_show=False,
                        type_="inside",
                        xaxis_index=[0, 1],
                        range_start=98,
                        range_end=100,
                    ),
                    opts.DataZoomOpts(
                        is_show=True,
                        xaxis_index=[0, 1],
                        type_="slider",
                        pos_top="85%",
                        range_start=98,
                        range_end=100,)
                        ,],)       
            )

        st_pyecharts(h,width="100%", height='500px')
