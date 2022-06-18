import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['zenhei']

#定义功能

# 选择搜索范围
def select_data(dataframe,keyword,platform): #dataframe , keyword:str, platform:str
    if platform == 'All':
        pass
    else:
        dataframe = dataframe[dataframe['平台类型'].isin([platform])]

    dataframe = dataframe[dataframe['关键词'].isin([keyword])]
    return dataframe.reset_index()

# 情感分析模块
def emotion_check(post):
    snownlp = SnowNLP(post)
    sentiments_score = snownlp.sentiments
    return sentiments_score

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
        #打印结果
        print(f'关键词:{mulitiple_word} \t 数量：{count} \t 占比: {(count/posts_length) * 100 :.2f}%')

    #AND 逻辑
    for mulitiple_word in ls2:
        #把有&拆分
        single = mulitiple_word.split('&')
        #遍历每个拆分后的词
        word_count = 0
        for word in single: 
            #更改 搜索列  ex:标题+内容
            #遍历每个 标题+内容列
            for post in content['标题+内容']:  # 更改excel表格对应列名
                #如果含有关键词 并辅助列=0
                try:
                    if word in post  :
                        #辅助列标记为1
                        word_count +=1
                except:
                    print(f'请检查excel表格单元格: {marker_length}行')
                    continue
            #取标记为1的sum
        d[mulitiple_word]  = word_count
        #打印结果
        print(f'关键词:{mulitiple_word} \t 数量：{word_count} \t 占比: {(word_count/posts_length) * 100 :.2f}%')

    print('*' * 50)
    sorted_d = dict(sorted(d.items(), key=lambda x: x[1],reverse =False))
    df = pd.DataFrame(list(sorted_d.items()) ,columns=['关键词','发文数量'])
    df.loc[:,'占比%'] =df['发文数量'] /posts_length * 100 
    #占比 保留 1位小数 %
    df.round({'占比%' : 1})


    #图像大小 figsize
    fig, ax = plt.subplots(figsize = (13,8)) 

    #数据 x,y轴
    x = list(sorted_d.keys())
    y = list(sorted_d.values())
    #柱状图
    ax.barh(x, y, height=0.6,alpha=0.8,fill=True,color = '#FFDAB9') 

    #size 调整字体大小
    plt.yticks(size=20,fontproperties = 'SimHei')
    #size 调整字体大小
    plt.xticks(size=20,fontproperties = 'SimHei')
    plt.title('发文数数量',fontproperties = 'SimHei' ,size=20)
    #标记y轴数量
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.2, 
                 str(round((i.get_width()), 2)), 
                 fontsize=16, fontweight='bold', 
                 color='grey') 
    #plt.show()
    figure = st.pyplot(fig=fig, clear_figure=None)
    return figure , df 

st.title('🌎Excel小工具')
uploaded_file = st.file_uploader(label="上传Excel文件" , type = ['csv','xlsx','xls'] )
col1 , col2, col3 = st.columns(3)
if uploaded_file is not None:

    if str(uploaded_file.type).split('/')[1] =='csv':
        dataframe = pd.read_csv(uploaded_file)
    else:
        dataframe = pd.read_excel(uploaded_file,dtype = str,index_col = False)

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
            plot,df= main(user_input,dataframe)
        # 渲染
        except:
            st.write('请检查Excel表格列名')

        #st.pyplot(fig=plot, clear_figure=None)
        st.dataframe(df)



with st.sidebar:
    st.subheader('🌟小功能使用步骤') 
    st.write('1: 传入Excel 文件')
    st.write('2: 选中对应平台/关键词')
    st.write('3: 输入关键词+Enter') 

    # image = Image.open('C:/Users/HFY/Desktop/streamlit/11.jpeg')
    # st.image(image, caption='')
