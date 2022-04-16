# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 21:08:34 2022

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 12:35:00 2022
@author: liamcassady
"""


#%% imports
import pandas as pd
from matplotlib.pyplot import hist
import matplotlib.pyplot as plt
from urllib.parse import urlparse


#%% read in data
aiml_data = pd.read_csv('reddit_database.csv')
aiml_data.head()


#%% basic info
aiml_data.info()
aiml_data.describe()
aiml_data['author_created_date'] = pd.to_datetime(aiml_data['author_created_utc'], unit='s')
aiml_data['created_date'] = pd.to_datetime(aiml_data['created_date'])
aiml_data.info()
aiml_data.describe(include='all', datetime_is_numeric=True)


#%% 1.1
Most_posts = aiml_data['subreddit'].str.lower().str.split().explode().value_counts()
print(Most_posts.head(5))


#%% 1.2
Most_posts_author = aiml_data['author'].str.lower().str.split().explode().value_counts()
print(Most_posts_author.head(5))


#%% 1.3
a = aiml_data.groupby('subreddit')['author'].describe()
print(a.sort_values(by='unique', ascending=False).head(5))


#%% 1.4
a = aiml_data.groupby('subreddit')['post'].describe(include=all)
print(a.sort_values(by='unique', ascending=False))


#%% 2.1
aiml_data.groupby('subreddit')['post'].count().plot(kind='bar')


#%% 2.2
a = aiml_data['score'].plot(kind='hist', range=[0,20])
a.set(xlabel="Scores", ylabel="Posts", title="Distribution of Post Scores")
a.get_figure()


#%% 2.3
aiml_data['dow'] = aiml_data['created_date'].dt.day_name()
aiml_data.groupby('dow')['created_date'].count().plot(kind='bar')
aiml_data['dow'] = pd.Categorical(aiml_data['dow'], categories=
    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday'],
    ordered=True)
dow_plot = aiml_data.groupby('dow')['created_date'].count().plot(kind='bar')
dow_plot.set(xlabel="Day Of The Week", ylabel="Total Number of Posts",
             title="AI/ML Posts Per day of the week")
dow_plot.get_figure()


#%% 2.4
aiml_data['hour'] = aiml_data['created_date'].dt.hour
a = aiml_data.groupby('hour')['created_date'].count().plot(kind='bar')
a.set(xlabel="Hour of The Day", ylabel="Posts",
             title="Average Posts Per Hour")
a.get_figure()


#%% 3.1
from datetime import datetime, timedelta
lastdayfrom = pd.to_datetime('6/8/2021')
aiml_data['dates'] = aiml_data['author_created_date'].dt.date
L = aiml_data.sort_index(ascending=False)
I = L[L["created_date"] >= (pd.to_datetime(lastdayfrom) - pd.Timedelta(days=30))]
print(I.groupby('subreddit').describe().head(5))


#%% 3.2
aiml_data['title_words'] = aiml_data['title'].str.count(' ')
a = aiml_data.groupby('title_words')['score'].count().plot()
a.set(xlabel="Title Word Count", ylabel="Scores",
             title="Scores Correlated Title Word Count")
a.get_figure()


#%% 3.3
title_words = aiml_data['title'].str.lower().str.split(expand=True).stack().value_counts()
print(title_words.head(20))


#%% 3.4 part1
def find_domains(text):
    domain = urlparse(text).netloc
    return domain
aiml_data['post_domains'] = aiml_data['title'].apply(find_domains)
x = aiml_data.groupby('post_domains').count()
print(x.sort_values(by='title', ascending=False).head(10))
