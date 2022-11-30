import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')
from datetime import date
import pandas_ta as ta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import time
st.set_page_config(page_title = '📈 AI Trading System',layout = 'wide')



BYD = yf.Ticker("1211.HK")
data = BYD.history(interval = "5m")
data['Datetime'] = data.index

def get_wr(high, low, close, lookback):
    highh = high.rolling(lookback).max() 
    lowl = low.rolling(lookback).min()
    wr = -100 * ((highh - close) / (highh - lowl))
    return wr


def pre_process(data):
    data.ta.rsi(close='Close', length=15, append=True, signal_indicators=True)
    data.ta.rsi(close='Close', length=25, append=True, signal_indicators=True)
    data.ta.rsi(close='Close', length=35, append=True, signal_indicators=True)
    data['wr15'] = get_wr(data['High'],data['Low'],data['Close'],15)
    data['wr25'] = get_wr(data['High'],data['Low'],data['Close'],25)
    data['wr35'] = get_wr(data['High'],data['Low'],data['Close'],35)
    data['atr15'] = ta.atr(data.High,data.Low,data.Close,window=15,fillna=False)
    data['atr25'] = ta.atr(data.High,data.Low,data.Close,window=25,fillna=False)
    data['atr35'] = ta.atr(data.High,data.Low,data.Close,window=35,fillna=False)
    data['sma15'] = data['Close'].rolling(15).mean()
    data['sma25'] = data['Close'].rolling(25).mean()
    data['sma35'] = data['Close'].rolling(35).mean()
    data.ta.stoch(high=data.High, low=data.Low, k=14, d=3, append=True)
    return data

def stock_price_visualize(data,stock_name):
    x = [i for i in range(data.shape[0])]

    # moving average 200
    MA200 = data['Close'].rolling(window =200).mean()
    # moving average 100
    MA100 = data['Close'].rolling(window =100).mean()
    # moving average 14
    MA14 = data['Close'].rolling(window =14).mean()

    fig = go.Figure(data=[go.Candlestick(x=x,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'])])

    # fig.layout = dict(xaxis=dict(type="category"))
    fig.add_trace(go.Scatter(x=x, y= MA200, mode='lines', line=dict(width = 1.5),marker_color = 'red',
                            showlegend=True, name = '长均线'))

    fig.add_trace(go.Scatter(x=x, y= MA100, mode='lines', line=dict(width = 1.5),marker_color = 'green', 
                            showlegend=True, name = '中均线'))

    fig.add_trace(go.Scatter(x=x, y= MA14,mode='lines', line=dict(width = 1.5),marker_color = 'yellow',
                            showlegend=True, name = '短均线'))

    layout = go.Layout(
        title = f'{stock_name}-K线图走势',
        plot_bgcolor='#efefef',
        # Font Families
        font_family='Monospace',
        font_color='#000000',
        font_size=18,
        xaxis=dict(
            rangeslider=dict(
                visible=False
            )
        )
    )
    fig.update_layout(layout)

    # fig.show()
    return fig

def RSI_plot(data):
    data.ta.rsi(close='Close', length=14, append=True, signal_indicators=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.25, 0.75])
    x = [i for i in range(data.shape[0])]
    fig.add_trace(go.Candlestick(
        x=x,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='#ff9900',
        decreasing_line_color='black',
        showlegend=False
    ))

    # Make RSI Plot
    fig.add_trace(go.Scatter(
        x=x,
        y=data['RSI_14'],
        line=dict(color='#ff9900', width=2),
        showlegend=False,
    ), row=2, col=1
    )
    # Add upper/lower bounds
    fig.update_yaxes(range=[-10, 110], row=2, col=1)
    fig.add_hline(y=0, col=1, row=2, line_color="#666", line_width=2)
    fig.add_hline(y=100, col=1, row=2, line_color="#666", line_width=2)

    # Add overbought/oversold
    fig.add_hline(y=30, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')
    fig.add_hline(y=70, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')

    # Customize font, colors, hide range slider
    layout = go.Layout(
        title = 'RSI14交易策略',
        plot_bgcolor='#efefef',
        # Font Families
        font_family='Monospace',
        font_color='#000000',
        font_size=18,
        xaxis=dict(
            rangeslider=dict(
                visible=False
            )
        )
    )
    fig.update_layout(layout)
    # update and display
    fig.update_layout(layout)
    # fig.show()
    return fig

st.markdown('# 📈AI 智能交易系统')

# with open('C:\\Users\\HFY\\Desktop\\app\\time.html' , 'r', encoding = 'utf-8') as f:
#     st.components.v1.html(f"""{f.read()}""")
st.markdown('#### 通过检测Long Term Moving Average(长期移动均线)与Short Term Moving Average(短期移动均线)交叉的交易情况,')
st.markdown('#### 结合RSI,MACD,WR等11类技术指标对交易市场,使用RNN(循环神经网络),Transfomer(注意力机制)模型,对下一时刻的最高价/最低价进行预测,以及预测进场时机')

st.markdown('***原模型训练集为2018~2020年外汇市场M15货币数据***')

tab0, tab1, tab2, tab3= st.tabs(['数据','K线图', '技术指标','预测模型'])

with tab0:
    LABEL_DATA = st.button('标记数据集')
    st.dataframe(data.iloc[::-1], width= 2000, height=600,use_container_width = False)
    if LABEL_DATA:
        data = pre_process(data)
        marker = st.success('完成标记技术指标数据', icon="✅")
        time.sleep(2)
        marker.empty()


with tab1:
    fig = stock_price_visualize(data,'比亚迪')
    st.plotly_chart(fig,use_container_width = True)

    st.markdown('#### 假如RSI处于超卖区域并开始上穿30水平,则你需要寻找看涨的反转烛台形态。此时为买入信号')
    st.markdown('#### 如果RSI处于超买区域并开始下穿70水平,则需要开始观察寻找看跌反转烛台。此时为做空信号')

    rsi_fig = RSI_plot(data)
    st.plotly_chart(rsi_fig,use_container_width = True)

with tab2:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(str(data['Datetime'][-2])[11:-6] + " 开盘价", str(data.Open[-2])[0:7], str(data.Open[-2] -data.Open[-3])[0:7])
    col2.metric(str(data['Datetime'][-2])[11:-6] + " 收盘价", str(data.Close[-2])[0:7],  str(data.Close[-2] -data.Close[-3])[0:7])
    col3.metric(str(data['Datetime'][-2])[11:-6] + " 最高价", str(data.High[-2])[0:7], str(data.High[-2] -data.High[-3])[0:7])
    col4.metric(str(data['Datetime'][-2])[11:-6] + " 最低价", str(data.Low[-2])[0:7], str(data.Low[-2] -data.Low[-3])[0:7])
    
    if LABEL_DATA:
        col1.metric(str(data['Datetime'][-2])[11:-6] + " RSI14", str(data.RSI_14[-2])[0:7], str(data.RSI_14[-2] - data.RSI_14[-3])[0:7])
        col2.metric(str(data['Datetime'][-2])[11:-6] + " WR15", str(data.wr15[-2])[0:7],  str(data.wr15[-2] - data.wr15[-3])[0:7])
        col3.metric(str(data['Datetime'][-2])[11:-6] + " ATR15", str(data.atr15[-2])[0:7], str(data.atr15[-2] -data.atr15[-3])[0:7])
        col4.metric(str(data['Datetime'][-2])[11:-6] + " SMA15", str(data.sma15[-2])[0:7], str(data.sma15[-2] -data.sma15[-3])[0:7])
