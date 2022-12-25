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
import requests
import json


# pyechart tutorial
#https://www.heywhale.com/mw/project/5eb7958f366f4d002d783d4a
st.set_page_config(page_title = '📈 AI Guided Trading System',layout = 'wide')

WINDOW_SIZE = 10
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown('# 📈AI Guided Financial Trading Dashboard')

st.markdown('#### 检测Long Term Moving Average(长期移动均线)与Short Term Moving Average(短期移动均线)')
st.markdown('#### RNN(循环神经网络),对下一时刻的最高价/最低价进行预测,以及预测进场时机')

st.markdown('##### >输入股票代号获取数据') 


###########################################################################
# Token accessToken 及权限校验机制
getAccessTokenUrl = 'https://quantapi.51ifind.com/api/v1/get_access_token'
# 获取refresh_token需下载Windows版本接口包解压，打开超级命令-工具-refresh_token查询
refreshtoken = 'eyJzaWduX3RpbWUiOiIyMDIyLTEyLTI1IDE5OjI1OjAxIn0=.eyJ1aWQiOiI2NjA1NzQ4MTYifQ==.770C980E4BFCAD438B549ADCFA5CF9AE6A9A2E2559F2E3174ADBF507C9A5D11E'
getAccessTokenHeader = {"Content- Type": "application/json", "refresh_token": refreshtoken}
getAccessTokenResponse = requests.post(url=getAccessTokenUrl, headers=getAccessTokenHeader)
accessToken = json.loads(getAccessTokenResponse.content)['data']['access_token']
print(accessToken)
thsHeaders = {"Content-Type": "application/json", "access_token": accessToken}
###########################################################################

###########################################################################
# handle data input / select perfer stock 
# STOCK = yf.Ticker('XPEV')
# if stock_name:
#     STOCK = yf.Ticker(stock_name)
# data = STOCK.history(interval = "15m")
# data['Datetime'] = data.index
# data['Datetime'] = data['Datetime'].astype(str)
########################################################################

button = st.button('刷新登录按钮')
stock_name = st.text_input('输入股票代号: ' , help = '查阅股票代号: https://finance.yahoo.com/lookup/',value = 'HC2301.SHF')
if button:
    F.login()


# data = F.handle_ifind_data(stock_name)
data = F.history_quotes(stock_name)
# convert to Asia timezone
# data['Datetime'] = pd.DataFrame(pd.to_datetime(data['Datetime'] ,utc=True).tz_convert('Asia/Shanghai')).index
data = F.pre_process(data,WINDOW_SIZE)

tab0, tab1, tab2, tab3= st.tabs(['数据','K线图', '技术指标','预测模型'])
with tab0:
    F.real_time()
    #st.dataframe(data, height=600,use_container_width = True)

with tab1:
    refresh = st.button('刷新K线图')
    if refresh:
        overlap_kline_line = F.draw_Kline(data,stock_name)
        # K线图 Echart
        st_pyecharts(overlap_kline_line,width="100%", height='900px')


with tab2:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(" 开盘价", str(data.Open.values[-1])[0:7], str(data.Open.values[-1] -data.Open.values[-2])[0:7])
    col2.metric(" 收盘价", str(data.Close.values[-1])[0:7],  str(data.Close.values[-1] -data.Close.values[-2])[0:7])
    col3.metric(" 最高价", str(data.High.values[-1])[0:7], str(data.High.values[-1] -data.High.values[-2])[0:7])
    col4.metric(" 最低价", str(data.Low.values[-1])[0:7], str(data.Low.values[-1] -data.Low.values[-2])[0:7])
    col1.metric(" 交易量", str(data.volume.values[-1])[0:7], str(data.volume.values[-1] - data.volume.values[-2])[0:7])


    # col2.metric(" WR25", str(data.wr25.values[-2])[0:7],  str(data.wr25.values[-1] - data.wr25.values[-2])[0:7])
    # col3.metric(" ATR25", str(data.atr25.values[-2])[0:7], str(data.atr15.values[-1] -data.atr25.values[-2])[0:7])
    # col4.metric(" SMA25", str(data.sma25.values[-2])[0:7], str(data.sma15.values[-1] -data.sma25.values[-2])[0:7])
    st.markdown('<div> <hr> </div>',unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(" RSI15", str(data.RSI_15.values[-2])[0:7], str(data.RSI_15.values[-2] - data.RSI_15.values[-3])[0:7])
    col2.metric(" WR15", str(data.wr15.values[-2])[0:7],  str(data.wr15.values[-2] - data.wr15.values[-3])[0:7])
    col3.metric(" ATR15", str(data.atr15.values[-2])[0:7], str(data.atr15.values[-2] -data.atr15.values[-3])[0:7])
    col4.metric( " SMA15", str(data.sma15.values[-2])[0:7], str(data.sma15.values[-2] -data.sma15.values[-3])[0:7])

with tab3:
    st.markdown('### 模型特征: ')
    st.dataframe(data)

    LABEL_MODEL = st.button('RNN模型预测')

    # file path "//app//wby//RNN.h5"
    model = keras.models.load_model("//app//wby//Bi_RNN.h5", compile=False)

    if LABEL_MODEL :
        with st.spinner(text="##### 正在处理数据..."):
            data = F.pre_process(data,WINDOW_SIZE)
            train_x_dict, price_scaler_max,price_scaler_min = F.generate_sequence(data,WINDOW_SIZE)
            predicted_max,predicted_min,predicted_label = F.make_prediction(model,train_x_dict,price_scaler_min,price_scaler_max)
            st.success('🚩已完成')
            
        # check model performance
        max_chart_data = pd.DataFrame({'预测最高值':[float(i) for i in predicted_max] , '真实最高值':data[f'max_{WINDOW_SIZE}'].tolist()[:len(data) - WINDOW_SIZE+1]})
        min_chart_data = pd.DataFrame({'预测最低值':[float(i) for i in predicted_min] , '真实最低值':data[f'min_{WINDOW_SIZE}'].tolist()[:len(data) - WINDOW_SIZE+1]})
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('### 预测最高值验证:')
            st.line_chart(max_chart_data, use_container_width = True)
        with col2:
            st.markdown('### 预测最低值验证:')
            st.line_chart(min_chart_data, use_container_width = True)
        
        
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
