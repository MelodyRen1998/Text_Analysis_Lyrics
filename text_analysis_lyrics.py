# -*- coding: utf-8 -*-
"""
Created on 2019.01.11 12:32:49

@author: yimeng
"""

import requests
import json
import pymongo
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
import os
from os import path
import numpy as np
import jieba
import wordcloud as wc
from PIL import Image
from matplotlib import pyplot as plt
from scipy.misc import imread
import random
from collections import Counter
import string
import jieba.analyse
import gensim
from gensim.models import KeyedVectors
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim import corpora, models, similarities


## 1. 抓取歌词数据
def main(page):
    print(page)
    url = 'https://c.y.qq.com/soso/fcgi-bin/client_search_cp'  # QQ音乐搜索链接

    data = {'qqmusic_ver': 1298,
            'remoteplace': 'txt.yqq.lyric',
            'inCharset': 'utf8',
            'sem': 1, 'ct': 24, 'catZhida': 1, 'p': page,
            'needNewCode': 0, 'platform': 'yqq',
            'lossless': 0, 'notice': 0, 'format': 'jsonp', 'outCharset': 'utf-8', 'loginUin': 0,
            'jsonpCallback': 'MusicJsonCallback4663515329750706',  # js请求
            'searchid': '107996317060494975',
            'hostUin': 0, 'n': 10, 'g_tk': 5381, 't': 7,
            'w': '陈奕迅', 'aggr': 0
            }

    headers = {'content-type': 'application/json',
               'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0'}
    r = requests.get(url, params=data, headers=headers)
    time.sleep(3)
    # 截取出现第一个歌词的字到最后一个
    response_str = r.text
    start = response_str.find("{")
    text = response_str[start:-1]
    #     print(text)
    result = json.loads(text)
    #     print(result)
    if result['code'] == 0:
        for info in result['data']['lyric']['list']:
            item = info['content']
            song_id = info['songmid']  # 找出歌曲id
            song_url = 'https://y.qq.com/n/yqq/song/' + song_id + '.html'  # 制作歌曲页面链接
            my_lyrics.append(item)
            song_urls.append(song_url)

my_lyrics = []
song_urls = []
for i in range(1, 21):
    main(i)

## 2. 抓取每首歌曲的发行时间
publish_date = []
for song in song_urls:
    headers = {'content-type': 'application/json',
               'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0'}
    r = requests.get(song, headers = headers)
    html_text = r.text
    song_start = 'song_detail.init({\r\n\t\t\r\n\t\t\tinfo : '  # 定位歌曲信息位置
    song_end = '\r\n\t\t\r\n\t\t});'
    index = html_text.find(song_start)
    index2 = html_text.find(song_end)
    song_info = html_text[index + len(song_start): index2]  # 提取歌曲信息
    mysong_info = json.loads(song_info)
    if 'pub_time' in mysong_info.keys():  # 判断是否有发行时间
        pub_date = mysong_info['pub_time']['content'][0]['value']
        publish_date.append(pub_date)
    else:
        publish_date.append('0000-00-00')

## 3. 分词
words = []  # 定义list
for lyrics in my_lyrics:  # 对文本文件逐行进行分词
    words = words + jieba.lcut(lyrics)  # 分词返回list，储存在words中
# 创建停用词list，对分好的词进行筛选
text = []
stop_path = input('Your stopword path is: ')  # E:\\renyimeng\\course\\stopwords.txt
with open(stop_path, 'rb') as f:  # 只读模式打开
    for line in f.readlines():  # 逐行读取
        text = text + line.decode().split()  # 读取后直接设置编码&按照空格分词

mywords = []
for word in words:  # 逐个筛选词语
    if word not in text:
        mywords.append(word)

## 4. 统计词频
word_count = Counter(mywords)  # 统计词频返回list
word_dict = dict(word_count)  # 转换为字典格式
# 搜集常见的中英文标点
zh_punctuations = '！？。“”＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕——'
en_punctuations = '''!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'''
punctuations = zh_punctuations + en_punctuations  # 所有的中英文标点
# 去除word_dict 里的标点符号
word_count_no_punctuations = [(word, freq) for word, freq in word_dict.items() if word not in punctuations]  # 去除标点
word_count_long = [(word, freq) for word, freq in word_count_no_punctuations if len(word) > 1]  # 选择长度>=2的
word_hanzi = [(word, freq) for word, freq in word_count_long if word[0] not in string.ascii_letters]  # 选择汉字词语
myfinal_word = [(word, freq) for word, freq in word_hanzi if word[0] not in string.digits]  # 选择汉字词语
word_sort = sorted(myfinal_word, key=lambda x: -x[1])  # 按照词频倒序排序
# 选择词频在top300的词语
word_sort_200 = word_sort[0:300]
print(word_sort_200)

## 5. 绘制词云
word_dict = dict(word_sort_200)  # 转换为字典
font = 'NotoSerifCJKsc-SemiBold.otf'  # 词云的字体文件
background_Image = np.array(Image.open("E:\\renyimeng\\course\\guitar.png"))  # 词云底图路径
img_colors = wc.ImageColorGenerator(background_Image)  # 提取底图颜色，作为词云颜色
cloud = wc.WordCloud(
    font_path = font,
    mask = background_Image,
    scale=2,
    max_font_size = 350,  # 最大字体
    min_font_size = 30,  # 最小字体
    max_words = 300,  # 词语的数量上限
    background_color = 'white').generate_from_frequencies(word_dict)
cloud.recolor(color_func = img_colors)
plt.figure()
plt.imshow(cloud,interpolation='bilinear')
plt.axis("off")
cloud.to_file('E:\\renyimeng\\course\\word-cloud.png')  # 把词云图保存下来

## 6. 绘制各年份发行歌曲数柱状图
pub_year = []
for date in publish_date:
    year = date[0:4]
    pub_year.append(year)

year_count = Counter(pub_year)  # 统计词频返回list
year_dict = dict(year_count)  # 转换为字典格式
my_year = [(word, freq) for word, freq in year_dict.items() if word not in ['0000']]  # 选择不为0000的发行时间
year_sort = sorted(my_year, key=lambda x: -x[1])  # 按照发行频数倒序排序
# print(year_sort)
year_freq = []
year_list = []
for year, freq in year_sort:
    year_freq.append(freq)
    year_list.append(year)

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.barh(range(len(year_freq)), year_freq, color = 'orange', tick_label = year_list)
plt.ylabel('发行年份')
plt.xlabel('歌曲数量')
savefig('E:\\renyimeng\\course\\pub_year.png')  # 保存绘制图片
plt.show()

## 7. 根据发行年份划分分析
# 每首歌分词 + 年份
lyric_word_year = list(zip(word_process, pub_year_int))
print(lyric_word_year[0])

before = [(word, year) for word, year in lyric_word_year if year < 2009]
after = [(word, year) for word, year in lyric_word_year if year >= 2009]
print(len(before))
print(len(after))

before_words = []
for (words, year) in before:
    before_words = before_words + words

after_words = []
for (words, year) in after:
    after_words = after_words + words

# 分别统计词频
before_count = Counter(before_words)  # 统计词频返回list
before_dict = dict(before_count)
before_count_long = [(word, freq) for word, freq in before_dict.items() if len(word) > 1]  # 选择长度>=2的
before_sort = sorted(before_count_long, key=lambda x: -x[1])  # 按照词频倒序排序
# 选择词频在top300的词语
before_sort_300 = before_sort[0:300]

after_count = Counter(after_words)  # 统计词频返回list
after_dict = dict(after_count)
after_count_long = [(word, freq) for word, freq in after_dict.items() if len(word) > 1]  # 选择长度>=2的
after_sort = sorted(after_count_long, key=lambda x: -x[1])  # 按照词频倒序排序
# 选择词频在top300的词语
after_sort_300 = after_sort[0:300]

# 分别绘制词云
word_dict = dict(before_sort_300)  # 转换为字典
font = 'NotoSerifCJKsc-SemiBold.otf'  # 词云的字体文件
background_Image = np.array(Image.open("E:\\renyimeng\\course\\word_pic\\pic2.jpg"))  # 词云底图路径
img_colors = wc.ImageColorGenerator(background_Image)  # 提取底图颜色，作为词云颜色
cloud = wc.WordCloud(
    font_path = font,
    mask = background_Image,
    scale=2,
    max_font_size = 40,  # 最大字体
    min_font_size = 1,  # 最小字体
    max_words = 300,  # 词语的数量上限
    background_color = 'white').generate_from_frequencies(word_dict)
cloud.recolor(color_func = img_colors)
plt.figure()
plt.imshow(cloud,interpolation='bilinear')
plt.axis("off")
cloud.to_file('E:\\renyimeng\\course\\before.png')  # 把词云图保存下来

# 分别绘制词云
word_dict = dict(after_sort_300)  # 转换为字典
font = 'NotoSerifCJKsc-SemiBold.otf'  # 词云的字体文件
background_Image = np.array(Image.open("E:\\renyimeng\\course\\word_pic\\pic5.jpg"))  # 词云底图路径
img_colors = wc.ImageColorGenerator(background_Image)  # 提取底图颜色，作为词云颜色
cloud = wc.WordCloud(
    font_path = font,
    mask = background_Image,
    scale=2,
    max_font_size = 40,  # 最大字体
    min_font_size = 1,  # 最小字体
    max_words = 300,  # 词语的数量上限
    background_color = 'white').generate_from_frequencies(word_dict)
cloud.recolor(color_func = img_colors)
plt.figure()
plt.imshow(cloud,interpolation='bilinear')
plt.axis("off")
cloud.to_file('E:\\renyimeng\\course\\after.png')  # 把词云图保存下来

## 8. 关键词提取
# 每首歌原始歌词 + 年份
lyric_year = list(zip(my_lyrics, pub_year_int))
before = [(word, year) for word, year in lyric_year if year < 2009]
after = [(word, year) for word, year in lyric_year if year >= 2009]

before_lyrics = ''
for lyric, year in before:
    before_lyrics = before_lyrics + lyric
after_lyrics = ''
for lyric, year in after:
    after_lyrics = after_lyrics + lyric
jieba.analyse.extract_tags(before_lyrics, topK = 5, withWeight = True, allowPOS = ('n'))

# 不同词性关键词
all_lyrics = ''
for lyric, year in lyric_year:
    all_lyrics = all_lyrics + lyric
print(jieba.analyse.extract_tags(before_lyrics, topK = 10, withWeight = True, allowPOS = ('n')))
print(jieba.analyse.extract_tags(after_lyrics, topK = 10, withWeight = True, allowPOS = ('n')))
print(jieba.analyse.extract_tags(all_lyrics, topK = 10, withWeight = True, allowPOS = ('v')))
print(jieba.analyse.extract_tags(all_lyrics, topK = 10, withWeight = True, allowPOS = ('a')))

# 动词词频统计
keyword = jieba.posseg.lcut(all_lyrics)
verb = []
for word, flag in keyword:
    if flag[0] == 'v':
        print(word)
        verb.append(word)

# 动词词频统计
verb_count = Counter(verb)  # 统计词频返回list
verb_dict = dict(verb_count)
verb_count_long = [(word, freq) for word, freq in verb_dict.items() if len(word) > 1]  # 选择长度>=2的
verb_sort = sorted(verb_count_long, key=lambda x: -x[1])  # 按照词频倒序排序
# 选择词频在top300的词语
verb_sort_300 = verb_sort[0:300]

# 绘制动词词云
word_dict = dict(verb_sort_300)  # 转换为字典
font = 'NotoSerifCJKsc-SemiBold.otf'  # 词云的字体文件
background_Image = np.array(Image.open("E:\\renyimeng\\course\\word_pic\\pic3.jpg"))  # 词云底图路径
img_colors = wc.ImageColorGenerator(background_Image)  # 提取底图颜色，作为词云颜色
cloud = wc.WordCloud(
    font_path = font,
    mask = background_Image,
    scale=2,
    max_font_size = 40,  # 最大字体
    min_font_size = 1,  # 最小字体
    max_words = 300,  # 词语的数量上限
    background_color = 'white').generate_from_frequencies(word_dict)
cloud.recolor(color_func = img_colors)
plt.figure()
plt.imshow(cloud,interpolation='bilinear')
plt.axis("off")
cloud.to_file('E:\\renyimeng\\course\\verb.png')  # 把词云图保存下来

# 形容词词频统计
keyword = jieba.posseg.lcut(all_lyrics)
adj = []
for word, flag in keyword:
    if flag[0] == 'a':
        print(word)
        adj.append(word)

verb_count = Counter(adj)  # 统计词频返回list
verb_dict = dict(verb_count)
verb_count_long = [(word, freq) for word, freq in verb_dict.items() if len(word) > 1]  # 选择长度>=2的
verb_sort = sorted(verb_count_long, key=lambda x: -x[1])  # 按照词频倒序排序
verb_sort_300 = verb_sort[0:300]

# 形容词词云
word_dict = dict(verb_sort_300)  # 转换为字典
font = 'NotoSerifCJKsc-SemiBold.otf'  # 词云的字体文件
background_Image = np.array(Image.open("E:\\renyimeng\\course\\word_pic\\pic4.jpg"))  # 词云底图路径
img_colors = wc.ImageColorGenerator(background_Image)  # 提取底图颜色，作为词云颜色
cloud = wc.WordCloud(
    font_path = font,
    mask = background_Image,
    scale=2,
    max_font_size = 40,  # 最大字体
    min_font_size = 1,  # 最小字体
    max_words = 300,  # 词语的数量上限
    background_color = 'white').generate_from_frequencies(word_dict)
cloud.recolor(color_func = img_colors)
plt.figure()
plt.imshow(cloud,interpolation='bilinear')
plt.axis("off")
cloud.to_file('E:\\renyimeng\\course\\adj.png')  # 把词云图保存下来

## 9. 词向量训练
mywords = []
for sentence in my_lyrics:
    temp = jieba.lcut(sentence)
    mywords.append(temp)
word_process = []
for words in mywords:
    temp = []
    for word in words:
        if word not in punctuations:
            if len(word) > 1:
                if word[0] not in string.ascii_letters:
                    if word[0] not in string.digits:
                        if len(word) > 0:
                            if word not in text:
                                temp.append(word)
    if len(temp) > 0:
        word_process.append(temp)

model = gensim.models.Word2Vec(size=300, min_count=1)
model.build_vocab(word_process)
# 加载已有的词向量
PRE_TRAINED_WORD2VEC = 'E:\\renyimeng\\course\\data\\sgns.wiki.word.bz2'
wv_from_text = KeyedVectors.load_word2vec_format(PRE_TRAINED_WORD2VEC, binary=False)  # 仅使用歌词训练
# 重新训练综合模型
model.build_vocab([list(wv_from_text.vocab.keys())], update = True)
model.intersect_word2vec_format(PRE_TRAINED_WORD2VEC, binary = False, lockf = 1.0)
# 再训练
model.train(word_process, total_examples = model.corpus_count, epochs = model.epochs)
# 进行相关性比较
print(model.similarity('幸福','孤独'))  # 综合模型
print(wv_from_text.similarity('幸福','孤独'))  # 歌词模型
print(model.similarity('幸福','浪漫'))  # 综合模型
print(wv_from_text.similarity('幸福','浪漫'))  # 歌词模型
print(model.similarity('十年','爱情'))  # 综合模型
print(wv_from_text.similarity('十年','爱情'))  # 歌词模型

## 10. LDA主题建模
train = []
stopwords = open('E:\\renyimeng\\course\\stopwords.txt','r',encoding='utf8').readlines()
stopwords = [ w.strip() for w in stopwords ]
for line in my_lyrics:
    line = jieba.lcut(line.rstrip())
    train.append([ w for w in line if w not in stopwords and w not in punctuations and len(w) > 1 and w[0] not in string.ascii_letters and w[0] not in string.digits and w not in stopwords])
dictionary = corpora.Dictionary(train)
print(dictionary)

corpus = [ dictionary.doc2bow(text) for text in train ]
corpus_tfidf = models.TfidfModel(corpus)[corpus]
lda = models.LdaModel(corpus, num_topics = 5, id2word = dictionary)  # 设定主题数为5
lda.print_topics(5)

# 查看每首歌曲所属的主题
topicList = lda.print_topics(5)
i=0
for doc in corpus_tfidf:
    topics = lda.get_document_topics(doc)
    topic = 0
    max_weight=0
    for t in topics:
        # topics是这篇文章的所有topic
        if t[1]>max_weight:
            topic = t[0]  # 保存权重最大的话题id
            max_weight = t[1]
    print ( str(i) +"\t" +str(topic) + "".join(train[i]) +"\t" + topicList[topic][1]  +"\t"  +"\n")  # 输出为【粘连的句子】+这个句子相关的话题，粘连句子之前的index为它在训练集的位置
    i= i+1