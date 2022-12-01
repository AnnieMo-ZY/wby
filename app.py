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


st.set_page_config(page_title = '📈 AI Guided Trading System',layout = 'wide')

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
    MA_PRGY_Task(data)
    GRYP_IDX = {}
    for idx, value in enumerate( list(data.event.unique())):
        GRYP_IDX[value] = idx
    data['event'].replace(GRYP_IDX, inplace= True)
    data.replace({'' : 0}, inplace = True)
    data.reset_index(inplace = True,drop = True)
    data.dropna(subset=['ratio top','RSI_35'],inplace=True)
    return data

def compare(Close,Red,Green,Yellow):

    # This function use built-in function sort to speed up the sort process.
    # Sort these 4 values (MA200,MA100,MA14,Close) in order to correctly label them from top to bottom
    #price:close price, Red = MovingAvg200, Green = MovingAvg100, Yellow = MovingAvg14

    #price_list = {'Open':Open,'High':High,'Low':Low,'Close':Close, 'Red':Red, 'Green':Green,'Yellow':Yellow}
    price_list = {'Price':Close, 'Red':Red, 'Green':Green,'Yellow':Yellow}
    # sort the dictionary base on the dict.values
    sorted_price_list = (sorted(price_list.items(), key=lambda x: x[1],reverse =True))
    # eg sorted_price_list dict(("Price":196.820)  ('Red':196.820) ('Yellow':196.780) ('Green':196.750))
    
    label = ''
    for i in sorted_price_list:
        # use dict keys and the first capital letter for labelling
        label+=i[0][0]
    # print(sorted_price_list,label)
    return label

def distance(Low,High,Red,Green,Yellow):

    # This function calculate the distance (%) between 4 lines (MA200,MA100,MA14,Close)
    # use the sort function to sort them base on the dict.values  return (max, second, third, fourth)
    # then calculate the difference between them and divide by the range so the percentage values add up to 100%   
    #price:Low/High, Red = MovingAvg200, Green = MovingAvg100, Yellow = MovingAvg14
    
    # case when all MA is above the candlestick
    if Red >= High and Green >= High and Yellow >= High:
        price_list = {'Price':High, 'Red':Red, 'Green':Green,'Yellow':Yellow}
    # case when all MA is below the candlestick

    elif Red <= Low and Green <= Low and Yellow <= Low:
        price_list = {'Price':Low, 'Red':Red, 'Green':Green,'Yellow':Yellow}
    # case when MA inside the candlestick
    else:
        return 0,0,0,0,0,0
    
    # sort the dictionary base on the dict.values
    sorted_price_list = (sorted(price_list.items(), key=lambda x: x[1],reverse =True))
    # distance between the maximum and minimum
    range = sorted_price_list[0][1] - sorted_price_list[3][1]
    # distance between the maximum and the second (%)
    ratio_top = (sorted_price_list[0][1] - sorted_price_list[1][1]) / range 
    # distance between the second and the third (%)
    ratio_mid = (sorted_price_list[1][1] - sorted_price_list[2][1]) / range
    # distance between the third and the fourth (%)
    ratio_bottom = (sorted_price_list[2][1] - sorted_price_list[3][1]) / range
    
    # distance in absolute value
    absolute_top = (sorted_price_list[0][1] - sorted_price_list[1][1])
    absolute_mid = (sorted_price_list[1][1] - sorted_price_list[2][1])
    absolute_bottom = (sorted_price_list[2][1] - sorted_price_list[3][1])

    return ratio_top, ratio_mid, ratio_bottom, absolute_top, absolute_mid, absolute_bottom


def within_price_range(Low,High,MA):
    if Low <= MA <= High:
        return 1
    else:
        return 0


def MA_PRGY_Task(data):  

    # moving average 200
    MA200 = data['Close'].rolling(window =200).mean()
    # moving average 100
    MA100 = data['Close'].rolling(window =100).mean()
    # moving average 14
    MA14 = data['Close'].rolling(window =14).mean()

    # add 4 columns to the dataframe
    data['event'] = ''

    Ratio_top_ls = ['' for i in range(200)]
    Ratio_mid_ls = ['' for i in range(200)]
    Ratio_bottom_ls = ['' for i in range(200)]

    Absolute_top_ls = ['' for i in range(200)]
    Absolute_mid_ls = ['' for i in range(200)]
    Absolute_bottom_ls = ['' for i in range(200)]

    # columns if MA is within low and high return 1, not in range return 0
    R_ls = ['' for i in range(200)]
    G_ls = ['' for i in range(200)]
    Y_ls = ['' for i in range(200)]

    label_list = []
    for i in (range(200,len(data),1)):
        Price = data.Close[i]
        Low = data.Low[i]
        High = data.High[i]
        Red = MA200[i]
        Green = MA100[i]
        Yellow = MA14[i]
        # return event label
        label = compare(Price,Red,Green,Yellow)

        # return distance ratio and absolute values
        ratio_top, ratio_mid, ratio_bottom, absolute_top, absolute_mid, absolute_bottom = distance(Low,High,Red,Green,Yellow)

        R_ls.append(within_price_range(Low, High, Red))
        G_ls.append(within_price_range(Low, High, Green))
        Y_ls.append(within_price_range(Low, High, Yellow))

        Ratio_top_ls.append(ratio_top)
        Ratio_mid_ls.append(ratio_mid)
        Ratio_bottom_ls.append(ratio_bottom)

        Absolute_top_ls.append(absolute_top)
        Absolute_mid_ls.append(absolute_mid)
        Absolute_bottom_ls.append(absolute_bottom)
    
        # first label append to the list start at 200 
        if len(label_list) == 0:
            data['event'][200] = label
        label_list.append(label)

        if len(label_list) > 1:
            # label added if is different from previou one
            if label_list[-1] != label_list[-2] :
                data['event'][i] = label
             # label not be added if is same from previou one
            elif label_list[-1] == label_list[-2] :
                data['event'][i] = ''

    # concat the dataframe
    data['ratio top'] = Ratio_top_ls
    data['ratio mid'] = Ratio_mid_ls
    data['ratio bottom'] = Ratio_bottom_ls
    data['absolute top'] = Absolute_top_ls
    data['absolute mid'] = Absolute_mid_ls
    data['absolute bottom'] = Absolute_bottom_ls
    # check if MA inside the candlestick range (1: within candlesitck, 0: outside candlestick)
    data['R'] = R_ls
    data['G'] = G_ls
    data['Y'] = Y_ls


def generate_sequence(data, window_size):
    train_dt_ori, train_dt_scaled, target_minprice, target_maxprice, target_minp_scaled,\
    target_maxp_scaled, price_scaler_max, price_scaler_min = [], [], [], [], [], [], [], []
    train_dt_GRYP,train_dt_distance,ta_indicators,target_bhs,macd_1,macd_2 = [],[],[],[],[],[]

    scaler = MinMaxScaler()
    scaler_a = MinMaxScaler()
    scaler_b = MinMaxScaler()
    data.reset_index(inplace = True,drop = True)

    for index, row in data.iterrows(): 
        if index <= len(data)- window_size:
            # OHLC numerical original data
            train_dt_ori.append(data.loc[index:window_size-1+index, ['Open', 'High', 'Low', 'Close']].values)
            # GRYP Categorical
            train_dt_GRYP.append(data.loc[index:window_size-1+index, ['event']].values)
            # macd_1.append(data.loc[index:window_size-1+index, ['dir_change']].values)
            # macd_2.append(data.loc[index:window_size-1+index, ['two_peaks']].values)
            # numerical feature min max scale
            train_dt_distance.append(scaler_a.fit_transform(data.loc[index:window_size-1+index, ['absolute top','absolute mid','absolute bottom',
                                                                    'ratio top','ratio mid','ratio bottom']].values))
            # TA indicators numerical min max scale
            
            ta_indicators.append(scaler_b.fit_transform(data.loc[index:window_size-1+index, [
                                        'wr15', 'wr25','wr35','atr15','RSI_15','RSI_25','RSI_35',
                                        'atr25','atr35','STOCHk_14_3_3','STOCHd_14_3_3' ,'sma15','sma25',
                                        'sma35']].values))

            # MinMax scale for the given windows size
            train_dt_scaled.append(scaler.fit_transform(data.loc[index:window_size-1+index, ['Open', 'High', 'Low', 'Close']].values))
            # # Choose min_ws labels for predictions
            # tmp_minprice = data.loc[window_size-1+index, f'min_{window_size}'].tolist()
            # target_minprice.append(tmp_minprice)
            # # Choose min_ws labels for predictions
            # tmp_maxprice = data.loc[window_size-1+index, f'max_{window_size}'].tolist()
            # target_maxprice.append(tmp_maxprice)
            # # Choose bhs label for prediction
            # target_bhs.append(data.loc[window_size-1+index, [f'ws{window_size}_pt10_sl8']].values)
            # given ws max and mim of the sequences
            max_v = max(data.loc[index:window_size-1+index, ['Open', 'High', 'Low', 'Close']].max())
            min_v = min(data.loc[index:window_size-1+index, ['Open', 'High', 'Low', 'Close']].min())
            
            # target_minp_scaled.append((tmp_minprice-min_v)/(max_v-min_v))
            # target_maxp_scaled.append((tmp_maxprice-min_v)/(max_v-min_v))
            # save the minimum and maximum for inverse transform to orginal scale
            price_scaler_max.append(max_v)
            price_scaler_min.append(min_v)

    # np.array convert correct data types
    train_arr_ohlc_scaled = np.array(train_dt_scaled).astype('float32')
    train_arr_GRYP = np.array(train_dt_GRYP).astype('int64')

    train_arr_distance = np.array(train_dt_distance).astype('float32')
    ta_indicators =  np.array(ta_indicators).astype('float32')
    # target_minpArr_scaled = np.array(target_minp_scaled).astype('float32')
    # target_maxpArr_scaled = np.array(target_maxp_scaled).astype('float32') 
    # target_arr_bhs = np.array(target_bhs).astype('int64')

    # TrainSet Features
    train_x_dict = {
        'OHLC':train_arr_ohlc_scaled,'GRYP': train_arr_GRYP,'DISTANCE': train_arr_distance, "TAINDICATORS" : ta_indicators
    }
    # TrainSet Ylabel
    # train_y_dict = {'minp': target_minpArr_scaled, 'maxp': target_maxpArr_scaled, 'bhs':target_arr_bhs}
    
    return train_x_dict, price_scaler_max,price_scaler_min

def process_model_result(y_pred,price_scaler_min, price_scaler_max):
    LABEL_INDEX = {1:'买', 0:'持仓', 2:'卖'}
    predicted_max,predicted_min = [],[]
    for i in range(len(y_pred[0])):
        #print(price_scaler_max[i] - price_scaler_min[i], y_pred[0][i] , price_scaler_min[i] )
        predicted_max.append((y_pred[0][i] * (price_scaler_max[i] - price_scaler_min[i])) + price_scaler_min[i] )
        predicted_min.append((y_pred[1][i] * (price_scaler_max[i] - price_scaler_min[i])) + price_scaler_min[i] )
    predicted_label = y_pred[2].argmax(axis=-1).tolist()
    st.markdown(f'预测下一时刻最高价为: {predicted_max[-1][0]}')
    st.markdown(f'预测下一时刻最低价为: {predicted_min[-1][0]}')
    st.markdown(f'预测下一时刻进场时机: {LABEL_INDEX[predicted_label[-1]]}')
    

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
        y=data['RSI_15'],
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
        title = 'RSI15交易策略',
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

st.markdown('# 📈AI Guided Financial Trading Dashboard')
data = pre_process(data)
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
        marker = st.success('完成标记技术指标数据', icon="✅")
        time.sleep(0.5)
        marker.empty()


with tab1:
    fig = stock_price_visualize(data,'比亚迪')
    st.plotly_chart(fig,use_container_width = True)
    if LABEL_DATA:
        st.markdown('#### 假如RSI处于超卖区域并开始上穿30水平,则你需要寻找看涨的反转烛台形态。此时为买入信号')
        st.markdown('#### 如果RSI处于超买区域并开始下穿70水平,则需要开始观察寻找看跌反转烛台。此时为做空信号')

        rsi_fig = RSI_plot(data)
        st.plotly_chart(rsi_fig,use_container_width = True)

with tab2:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(str(data['Datetime'].values[-2])[11:-6] + " 开盘价", str(data.Open.values[-2])[0:7], str(data.Open.values[-2] -data.Open.values[-3])[0:7])
    col2.metric(str(data['Datetime'].values[-2])[11:-6] + " 收盘价", str(data.Close.values[-2])[0:7],  str(data.Close.values[-2] -data.Close.values[-3])[0:7])
    col3.metric(str(data['Datetime'].values[-2])[11:-6] + " 最高价", str(data.High.values[-2])[0:7], str(data.High.values[-2] -data.High.values[-3])[0:7])
    col4.metric(str(data['Datetime'].values[-2])[11:-6] + " 最低价", str(data.Low.values[-2])[0:7], str(data.Low.values[-2] -data.Low.values[-3])[0:7])
    
    if LABEL_DATA:
        col1.metric(str(data['Datetime'].values[-2])[11:-6] + " RSI15", str(data.RSI_15.values[-2])[0:7], str(data.RSI_15.values[-2] - data.RSI_15.values[-3])[0:7])
        col2.metric(str(data['Datetime'].values[-2])[11:-6] + " WR15", str(data.wr15.values[-2])[0:7],  str(data.wr15.values[-2] - data.wr15.values[-3])[0:7])
        col3.metric(str(data['Datetime'].values[-2])[11:-6] + " ATR15", str(data.atr15.values[-2])[0:7], str(data.atr15.values[-2] -data.atr15.values[-3])[0:7])
        col4.metric(str(data['Datetime'].values[-2])[11:-6] + " SMA15", str(data.sma15.values[-2])[0:7], str(data.sma15.values[-2] -data.sma15.values[-3])[0:7])

        col1.metric(str(data['Datetime'].values[-2])[11:-6] + " RSI25", str(data.RSI_25.values[-2])[0:7], str(data.RSI_25.values[-2] - data.RSI_25.values[-3])[0:7])
        col2.metric(str(data['Datetime'].values[-2])[11:-6] + " WR25", str(data.wr25.values[-2])[0:7],  str(data.wr25.values[-2] - data.wr25.values[-3])[0:7])
        col3.metric(str(data['Datetime'].values[-2])[11:-6] + " ATR25", str(data.atr25.values[-2])[0:7], str(data.atr15.values[-2] -data.atr25.values[-3])[0:7])
        col4.metric(str(data['Datetime'].values[-2])[11:-6] + " SMA25", str(data.sma25.values[-2])[0:7], str(data.sma15.values[-2] -data.sma25.values[-3])[0:7])


def make_prediction(model,train_x_dict, price_scaler_min,price_scaler_max):
    y_pred = model.predict(train_x_dict)
    process_model_result(y_pred, price_scaler_min,price_scaler_max)


with tab3:
    WINDOW_SIZE = 10
    
    st.markdown('### 模型特征: ')
    st.dataframe(data)
    # Train Set
    with st.spinner(text="##### 正在处理数据..."):
        train_x_dict, price_scaler_max,price_scaler_min = generate_sequence(data,WINDOW_SIZE)
    
    LABEL_MODEL = st.button('RNN模型预测')
    model = keras.models.load_model('.\\RNN.h5', compile=False)
    if LABEL_MODEL:
        make_prediction(model,train_x_dict,price_scaler_min,price_scaler_max)
        st.info('Finished')

            
