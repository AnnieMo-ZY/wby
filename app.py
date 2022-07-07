import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import time
from datetime import datetime

def count_down():
    today = datetime.today()
    year_month = str(today).split()[0]

    xiaban = datetime.strptime('{} 18:00:00'.format(year_month), '%Y-%m-%d %H:%M:%S')

    time_diff = str(xiaban - today)

    hour = int(time_diff.split(':')[0])-8
    min = time_diff.split(':')[1]
    sec = float(time_diff.split(':')[2])
    
    with st.empty():
        st.write('距离下班还要:{}小时 {}分钟 {:.2f}秒'.format(hour, min, sec))




def and_algo(post,single):
    count = 0 
    #print(len(single))
    for word  in single:
        #print(word)
        if word in post:
            count+=1
    if count == len(single):
        return 1
    else:
        return 0

def merge(df_ls):
    defined_columns = ["平台类型","identify_id",'分类',"media_id","media_url","标题","内容","文章创建时间"
                   ,"关键词","account_id","性别","地域","账号名称","简介","认证原因","主页链接","粉丝数","产品线","微闪投"
                  ,"上架状态","账号分类","短视频转发数","短视频评论数","短视频点赞数","是否命中"]
    files = []
    for df in df_ls:
        copy_df = df.copy() #使用备份 不在源数据更改 可重复运行
        for col in copy_df.columns:
        #更改列名
            if '是否微闪投' in col:
                copy_df.rename(columns={"是否微闪投" : "微闪投"}, inplace = True)
            if '是否是精确搜索' in col:
                copy_df.rename(columns={"是否是精确搜索" : "是否命中"}, inplace = True) 
            if 'red_id' in col:
                copy_df.rename(columns={"red_id" : "account_id"}, inplace = True)
            if '账号简介' in col:
                copy_df.rename(columns={"账号简介" : "简介"}, inplace = True)
            if '微博评论数' in col:
                copy_df.rename(columns={"微博评论数" : "短视频评论数",'微博转发数':'短视频转发数' ,'微博点赞数':'短视频点赞数' }, inplace = True)
            if '标题内容' in col:
                copy_df.rename(columns={'标题内容':'内容' }, inplace = True)
            if '是否上架' in col:
                copy_df.rename(columns={'是否上架':'上架状态' }, inplace = True)
                
       
        #不存在的列名补齐
        for d in defined_columns:
            if d not in list(copy_df.columns):
                copy_df.insert(copy_df.shape[1], column = d, value ='null')
        copy_df = copy_df[defined_columns]
        files.append(copy_df)
    final = pd.concat([pd.DataFrame(i) for i in files])
    return final

#定义功能

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(encoding = 'utf_8_sig', line_terminator="\n", index = False).encode('utf_8_sig')

# 选择搜索范围
def select_data(dataframe,keyword,platform): #dataframe , keyword:str, platform:str
    if platform == 'All':
        pass
    else:
        dataframe = dataframe[dataframe['平台类型'].isin([platform])]

    dataframe = dataframe[dataframe['关键词'].isin([keyword])]
    return dataframe.reset_index()

# 情感分析模块
# def emotion_check(post):
#     snownlp = SnowNLP(post)
#     sentiments_score = snownlp.sentiments
#     return sentiments_score

# 搜索逻辑模块
def main(user_input, dataframe):
    content  = dataframe
    posts_length = content.shape[0]
    word_list = user_input.strip().split(',')
    ls1 = [] # 保存含有 / 的词 
    ls2 = [] # 保存含有 & 和其他词
    for word in word_list:
        if '/' in word:
            ls1.append(word)
        else:
            ls2.append(word)
    d={}

    # FOR循环 遍历word_list的词
    for mulitiple_word in ls1:
        d.update({mulitiple_word:0})
        #把有/拆分
        single = mulitiple_word.split('/')
        #创建辅助列 =0
        content.loc[:,'辅助列']= 0
        #遍历每个拆分后的词
        for word in single: 
            marker_length = 0
            #更改 搜索列  ex:标题+内容
            #遍历每个 标题+内容列
            for post in content['标题+内容']:  # 更改excel表格对应列名
                #如果含有关键词 并辅助列=0
                try:
                    if word in post  and content['辅助列'][marker_length] == 0 :
                        #辅助列标记为1
                        content['辅助列'][marker_length] = 1
                    marker_length+=1
                except:
                    print(f'请检查excel表格单元格: {marker_length}行')
                    continue
            #取标记为1的sum
            count = content['辅助列'].sum()
        d[mulitiple_word]  = count
        #print(f'关键词:{mulitiple_word} \t 数量：{count} \t 占比: {(count/posts_length) * 100 :.2f}%')

    #AND 逻辑
    for mulitiple_word in ls2:
        #把有&拆分
        single = mulitiple_word.split('&')
        #遍历每个拆分后的词
        word_count = 0
        #temp 储存 and_algo结果
        temp = []
        for post in content['标题+内容']: 
            aa = and_algo(post, single)
            temp.append(aa)
            word_count = sum(temp) 
        d[mulitiple_word]  = word_count


    sorted_d = dict(sorted(d.items(), key=lambda x: x[1],reverse =False))
    df = pd.DataFrame(list(sorted_d.items()) ,columns=['关键词','发文数量'])
    df.loc[:,'占比%'] =df['发文数量'] /posts_length * 100 
    #占比 保留 1位小数 %
    df.round({'占比%' : 1})

    #数据 x,y轴
    x = list(sorted_d.keys())
    y = list(sorted_d.values())

    return x , y, df

st.title('🌎Excel小工具')
    
uploaded_file = st.file_uploader(label="上传Excel文件" , type = ['csv','xlsx'],accept_multiple_files=True )
time.sleep(1)
#合并文件
if len(uploaded_file) > 1:
    time.sleep(1)
    df_ls = []
    for index, item  in enumerate(uploaded_file):
        if str(item.name).split('.')[1] == 'csv':
            df = pd.read_csv(uploaded_file[index], encoding = 'utf-8') #encoding='gb18030'
            st.write('csv读取成功')
            df_ls.append(df)
        elif str(item.name).split('.')[1] == 'xlsx':
            df = pd.read_excel(uploaded_file[index])
            st.write('xlsx读取成功')
            df_ls.append(df)
        st.write(f'共{len(df_ls)}份文件')
            
     
    try:
        concat_data = merge(df_ls)
        concat_data.fillna({'粉丝数':0},inplace =True)
        concat_data.fillna('null',inplace =True)
        
    except Exception as e:
        st.write('检查列名')
        st.write(str(e))
    csv = convert_df(concat_data)

    st.download_button(
        label="下载合并文件 as CSV",
        data=csv,
        file_name='combined_file.csv',
        mime = 'text/csv')






col1 , col2, col3 = st.columns(3)
if len(uploaded_file) == 1:
    if str(uploaded_file[0].type).split('/')[1] =='csv':
        dataframe = pd.read_csv(uploaded_file[0])
    else:
        dataframe = pd.read_excel(uploaded_file[0],dtype = str,index_col = False)

    selected_keyword = list(dataframe['关键词'].unique())
    selected_platform = list(dataframe['平台类型'].unique())
    selected_platform.extend(['All'])
    with st.container():
        with col1:
            pltf = st.selectbox('选择平台', selected_platform)
        with col2:
            keyword = st.selectbox( '选择关键词', selected_keyword)
    dataframe = select_data(dataframe,keyword,pltf)
    st.write('已选择:', pltf,keyword)
    st.write('数据表')
    st.dataframe(dataframe,1000,100)
    user_input = st.text_input('输入搜索关键词', '')
# 结果
    if user_input:
        try:
            x,y,df =main(user_input,dataframe)

        except:
            st.write('请检查Excel表格列名')

        main_result_jason = {'关键词': x, '数量':[int(i) for i in y]}

        df= pd.DataFrame(main_result_jason)
        bars = alt.Chart(df).mark_bar().encode(
            y=alt.Y('关键词',sort = '-x'),
            x=alt.X('数量:Q' )).properties( height = 400)

        st.altair_chart(bars, use_container_width=True)
        st.dataframe(df)


with st.sidebar:
    st.subheader('🌟使用步骤') 
    st.write('1: 传入Excel 文件')
    st.write('2: 选中对应平台/关键词')
    st.write('3: 输入关键词+Enter')
    # image = Image.open('C:/Users/HFY/Desktop/streamlit/11.jpeg')
    # st.image(image, caption='')

    # if len(uploaded_file)>0:
    #     files_ls = [pd.read_excel(file) for file in uploaded_file]

    #     concat_data = pd.concat(files_ls,sort=True)

    #     csv = convert_df(concat_data)

    #     st.download_button(
    #         label="下载合并文件 as CSV",
    #         data=csv,
    #         file_name='combined_file.csv',)

    
with st.empty():
for i in range(0,100):
    count_down()
    i+=1
    time.sleep(1)
