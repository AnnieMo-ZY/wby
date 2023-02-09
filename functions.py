import streamlit as st
import numpy as np
import pandas as pd
import warnings
from tensorflow import keras
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
# new
from pyecharts.charts import *
from pyecharts import options as opts
from streamlit_echarts import st_pyecharts
import tushare as ts
pro = ts.pro_api('8800190d8a7e7403c41b4053294d5b289b41f7cd4f90acf81632790b')

def long_volume(stock_name):
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    df = pro.fut_holding(symbol=stock_name.split('.')[0],trade_date = yesterday)
    df.fillna(0,inplace=True)
    df.sort_values(by='long_hld',ascending=False,inplace=True)
    df2 = df.copy()
    df2.sort_values(by='short_hld',ascending=False,inplace=True)
    long = (
        Bar(
            init_opts=opts.InitOpts()
        )
        .add_xaxis(list(df['broker']))
        .add_yaxis("多头持仓量", list(df.long_hld.values),itemstyle_opts=opts.ItemStyleOpts(color='rgb(205,51,0)'))
        # .add_yaxis("空头持仓量", list(df.short_hld.values))
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="{} 多头持仓排名".format(stock_name),
                title_textstyle_opts=opts.TextStyleOpts(color="black"),
            )
        )
    )
    short =  (
        Bar(
            init_opts=opts.InitOpts()
        )
        .add_xaxis(list(df2['broker']))
        # .add_yaxis("多头持仓量", list(df.long_hld.values))
        .add_yaxis("空头持仓量", list(df2.short_hld.values),itemstyle_opts=opts.ItemStyleOpts(color='rgb(69,139,116)'))
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="{} 空头持仓排名".format(stock_name),
                title_textstyle_opts=opts.TextStyleOpts(color="black"),
            )
        )
    )
    return long,short

def rename_dataframe(data):
    data.rename(columns={'close' : 'Close', 'open':'Open' , 'high' : 'High' , 'low':'Low','vol':'volume'},errors='raise',inplace=True)
    data['Datetime'] = data.trade_date
    data = data.iloc[::-1]
    return data

def get_wr(high, low, close, lookback):
    highh = high.rolling(lookback).max() 
    lowl = low.rolling(lookback).min()
    wr = -100 * ((highh - close) / (highh - lowl))
    return wr

def make_prediction(model,train_x_dict, price_scaler_min,price_scaler_max):
    y_pred = model.predict(train_x_dict)
    predicted_max,predicted_min,predicted_label = process_model_result(y_pred, price_scaler_min,price_scaler_max)
    return predicted_max,predicted_min,predicted_label

def label_min_max(df, ws):
    local_min = np.array([])
    local_max = np.array([])
    for i in (range(len(df)-ws)):
        local_min = np.append(local_min, df.Low.iloc[i:i+ws].min()) 
        local_max = np.append(local_max, df.High.iloc[i:i+ws].max()) 
    for i in range(ws):
        local_min = np.append(local_min, df.Low.iloc[-ws:].min()) 
        local_max = np.append(local_max, df.High.iloc[-ws:].max()) 
    df[f'min_{ws}']=local_min
    df[f'max_{ws}']=local_max
    return df

def pre_process(data,window_size):
    data.reset_index(inplace = True)
    # calculate sma
    data['sma'] = data['Close'].rolling(20).mean()
    # calculate standard deviation
    data['sd'] = data['Close'].rolling(20).std()
    # calculate lower band
    data['lb'] = data['sma'] - 2 * data['sd']
    # calculate upper band
    data['ub'] = data['sma'] + 2 * data['sd']
    data.ta.rsi(close='Close', length=15, append=True, signal_indicators=True)
    data.ta.rsi(close='Close', length=25, append=True, signal_indicators=True)
    data.ta.rsi(close='Close', length=35, append=True, signal_indicators=True)
    data['wr15'] = get_wr(data['High'],data['Low'],data['Close'],15)
    data['wr25'] = get_wr(data['High'],data['Low'],data['Close'],25)
    data['wr35'] = get_wr(data['High'],data['Low'],data['Close'],35)
    data['atr15'] = ta.atr(data.High,data.Low,data.Close,window=15,fillna=0)
    data['atr25'] = ta.atr(data.High,data.Low,data.Close,window=25,fillna=0)
    data['atr35'] = ta.atr(data.High,data.Low,data.Close,window=35,fillna=0)
    data['sma15'] = data['Close'].rolling(15).mean()
    data['sma25'] = data['Close'].rolling(25).mean()
    data['sma35'] = data['Close'].rolling(35).mean()
    data.ta.stoch(high=data.High, low=data.Low, k=14, d=3, append=True)
    event(data,data.High,data.Low, data.lb, data.ub, data.sma,len(data.High))
    MA_PRGY_Task(data)
    GRYP_IDX = {}
    for idx, value in enumerate(list(data.event.unique())):
        GRYP_IDX[value] = idx
    data['event'].replace(GRYP_IDX, inplace= True)
    data = label_min_max(data,window_size)
    data.replace({'': 0}, inplace = True)
    data.fillna(0,inplace=True)
    # data.dropna(subset=['ratio top','RSI_35',f'min_{window_size}'],inplace=True)
    return data

# detect bb_event
def event(df,high,low, lower_band, upper_band, middle_band,l):

    # Outside the lower BB
    def event_1(high, lower_band):
        if high < lower_band:
            return 1
        else:
            return 0
    df['bb_event1'] = np.vectorize(event_1)(high, lower_band)

    # Outside the upper BB
    def event_2(low, upper_band):
        if low > upper_band:
            return 1
        else:
            return 0
    df['bb_event2'] = np.vectorize(event_2)(low, upper_band)

    # Touches the lower BB
    def event_3(high, lower_band):
        if 0.9999*lower_band < high < 1.0001*lower_band:
            return 1
        else:
            return 0
    df['bb_event3'] = np.vectorize(event_3)(high, lower_band)


    # Touches the upper BB
    def event_4(low, upper_band):
        if 0.9999*upper_band < low < 1.0001*upper_band:
            return 1
        else:
            return 0
    df['bb_event4'] = np.vectorize(event_4)(low, upper_band)

    # Touches the middle BB from Top
    def event_5(high, middle_band):
        if 0.9999*middle_band < high < 1.0001*middle_band:
            return 1
        else:
            return 0
    df['bb_event5'] = np.vectorize(event_5)(high, middle_band)

    # Touches the middle BB from Bottom
    def event_6(low, middle_band):
        if 0.9999*middle_band < low < 1.0001*middle_band:
            return 1
        else:
            return 0
    df['bb_event6'] = np.vectorize(event_6)(low, middle_band)

    # Crosses the middle BB from Top towards Bottom
    def event_7(high,low, middle_band,l): 
        if l-1<=19:
            return 0
        if low[l]<middle_band[l] and middle_band[l]<high[l] and middle_band[l+1]<middle_band[l] and middle_band[l-1]>middle_band[l]:
            return 1
        else:
            return 0
    df['bb_event7'] = df.apply(lambda l: event_7(high,low, middle_band,l.name), axis=1)

    # Crosses the middle BB from Bottom towards Top
    def event_8(high,low, middle_band,l): 
        if l-1<=19:
            return 0
        if low[l]<middle_band[l] and middle_band[l]<high[l] and middle_band[l+1]>middle_band[l] and middle_band[l-1]<middle_band[l]:
            return 1
        else:
            return 0
    df['bb_event8'] = df.apply(lambda l: event_8(high,low, middle_band,l.name), axis=1)
    

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

    # moving average 75
    MA75 = data['Close'].rolling(window =75).mean()
    # moving average 100
    MA100 = data['Close'].rolling(window =100).mean()
    # moving average 14
    MA14 = data['Close'].rolling(window =14).mean()

    # add 4 columns to the dataframe
    data['event'] = ''

    Ratio_top_ls = [0 for i in range(100)]
    Ratio_mid_ls = [0 for i in range(100)]
    Ratio_bottom_ls = [0 for i in range(100)]

    Absolute_top_ls = [0 for i in range(100)]
    Absolute_mid_ls = [0 for i in range(100)]
    Absolute_bottom_ls = [0 for i in range(100)]

    # columns if MA is within low and high return 1, not in range return 0
    R_ls = [0 for i in range(100)]
    G_ls = [0 for i in range(100)]
    Y_ls = [0 for i in range(100)]

    label_list = []
    for i in (range(100,len(data),1)):
        Price = data.Close[i]
        Low = data.Low[i]
        High = data.High[i]
        Red = MA75[i]
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
            data['event'][100] = label
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
    train_dt_ori, train_dt_scaled, price_scaler_max, price_scaler_min = [], [], [], []
    train_dt_GRYP,train_dt_distance,ta_indicators,r, g,y,bb_event = [],[],[],[],[],[],[]

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
            r.append(data.loc[index:window_size-1+index, ['R']].values)
            g.append(data.loc[index:window_size-1+index, ['G']].values)
            y.append(data.loc[index:window_size-1+index, ['Y']].values)
            bb_event.append(data.loc[index:window_size-1+index, ['bb_event1','bb_event2','bb_event3','bb_event4',
                                                        'bb_event5','bb_event6','bb_event7','bb_event8']].values)
            # macd_1.append(data.loc[index:window_size-1+index, ['dir_change']].values)

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
            # target_bhs.append(data.loc[window_size-1+index, [f'ws{window_size}_pt10_sl8']].values)
            # given ws max and mim of the sequences
            max_v = max(data.loc[index:window_size-1+index, ['Open', 'High', 'Low', 'Close']].max())
            min_v = min(data.loc[index:window_size-1+index, ['Open', 'High', 'Low', 'Close']].min())
            
            # save the minimum and maximum for inverse transform to orginal scale
            price_scaler_max.append(max_v)
            price_scaler_min.append(min_v)

    # np.array convert correct data types
    train_arr_ohlc_scaled = np.array(train_dt_scaled).astype('float32')
    train_arr_GRYP = np.array(train_dt_GRYP).astype('int64')
    r = np.array(r).astype('int64')
    g = np.array(g).astype('int64')
    y = np.array(y).astype('int64')
    bb_event = np.array(bb_event).astype('float32')
    train_arr_distance = np.array(train_dt_distance).astype('float32')
    ta_indicators =  np.array(ta_indicators).astype('float64')

    # TrainSet Features
    train_x_dict = {
        'OHLC':train_arr_ohlc_scaled,'GRYP': train_arr_GRYP,'DISTANCE': train_arr_distance, "TAINDICATORS" : ta_indicators,
        'R':r ,'G':g ,'Y':y , 'BB' : bb_event
    }

    # TrainSet Ylabel
    # train_y_dict = {'minp': target_minpArr_scaled, 'maxp': target_maxpArr_scaled, 'bhs':target_arr_bhs}
    
    return train_x_dict, price_scaler_max,price_scaler_min

def process_model_result(y_pred,price_scaler_min, price_scaler_max):
    LABEL_INDEX = {1:'做多', 0:'持仓', 2:'做空'}
    predicted_max,predicted_min = [],[]
    for i in range(len(y_pred[0])):
        #print(price_scaler_max[i] - price_scaler_min[i], y_pred[0][i] , price_scaler_min[i] )
        predicted_max.append((y_pred[0][i] * (price_scaler_max[i] - price_scaler_min[i])) + price_scaler_min[i] )
        predicted_min.append((y_pred[1][i] * (price_scaler_max[i] - price_scaler_min[i])) + price_scaler_min[i] )
    predicted_label = y_pred[2].argmax(axis=-1).tolist()
    st.markdown(f'***预测下一个周期最高价为: {predicted_max[-1][0]}***')
    st.markdown(f'***预测下一个周期低价为: {predicted_min[-1][0]}***')
    st.markdown(f'***预测下一个周期场时机: {LABEL_INDEX[predicted_label[-1]]}***')
    return predicted_max,predicted_min,predicted_label
    

def label_to_marker(data,predicted_label):
    marker_ls = []
    for i in range(len(predicted_label)):
        # label 1=buy 2=sell
        if predicted_label[i] == 1:
            marker_ls.append(opts.MarkPointItem(coord=[data['Datetime'].tolist()[i], data['Low'].tolist()[i] - 0.05], value="做多"))
        if predicted_label[i] == 2:
            marker_ls.append(opts.MarkPointItem(coord=[data['Datetime'].tolist()[i], data['High'].tolist()[i] + 0.05], value="做空"))
    return marker_ls


def draw_Kline(data,stock_name,cycle_select):
    MA70 = data['Close'].rolling(window =70).mean()
    # moving average 100
    MA100 = data['Close'].rolling(window =100).mean()
    # moving average 14
    MA14 = data['Close'].rolling(window =14).mean()

    line = (
            Line()
            .add_xaxis(xaxis_data=data["Datetime"])
            .add_yaxis(
                series_name="MA14",
                y_axis=MA14,
                is_smooth=True,
                is_hover_animation=False,
                linestyle_opts=opts.LineStyleOpts(width=3, opacity=0.5),
                label_opts=opts.LabelOpts(is_show=False),)
            .add_yaxis(
                series_name="MA100",
                y_axis=MA100,
                is_smooth=True,
                is_hover_animation=False,
                linestyle_opts=opts.LineStyleOpts(width=3, opacity=0.5),
                label_opts=opts.LabelOpts(is_show=False),)
            .add_yaxis(
                series_name="MA70",
                y_axis=MA70,
                is_smooth=True,
                is_hover_animation=True,
                linestyle_opts=opts.LineStyleOpts(width=3, opacity=0.5),
                label_opts=opts.LabelOpts(is_show=False),)
            )


    g = (Kline(init_opts=opts.InitOpts(width="900%", height='500%'))
            .add_xaxis(data['Datetime'].tolist()) 
            
            #y轴数据，默认open、close、low、high，转为list格式
            .add_yaxis("",y_axis=data[['Open', 'Close', 'Low', 'High']].values.tolist(),
            # 设置烛台颜色
            itemstyle_opts=opts.ItemStyleOpts(
            color="rgb(205,51,0)",#阳线红色 涨 #FF0000
            color0="rgb(69,139,116)",#阴线绿色 跌 #32CD32
            border_color="rgb(205,51,0)",
            border_color0="rgb(69,139,116)",),
            # 显示辅助线 均价
            markline_opts=opts.MarkLineOpts(
            data=[opts.MarkLineItem(name='平均价格',type_="average", value_dim='close')]))

            .set_global_opts(
            #标题
            title_opts =opts.TitleOpts(title = f'{stock_name} K线图',
            #副标题
            subtitle = cycle_select,pos_left = 'left',
            title_textstyle_opts = opts.TextStyleOpts(font_size=35),
            subtitle_textstyle_opts = opts.TextStyleOpts(font_size=28),),
            # 图例
            legend_opts=opts.LegendOpts(
                is_show=True, 
                pos_top=20,
                pos_left="center",item_width =30 ,item_height=25 ,
                textstyle_opts = opts.TextStyleOpts(font_size = 20)),
            #
            xaxis_opts=opts.AxisOpts(is_scale=True),
            yaxis_opts=opts.AxisOpts(is_scale=True,),

            # 浮动十字辅助线
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross",is_show_content=True),
            # 缩放
            datazoom_opts=[
                opts.DataZoomOpts(
                    is_show=False,
                    type_="inside",
                    xaxis_index=[0, 1],
                    range_start=95,
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

    overlap_kline_line = g.overlap(line)
    return overlap_kline_line
