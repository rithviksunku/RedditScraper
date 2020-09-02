#!/usr/bin/env python
# coding: utf-8

# In[6]:


get_ipython().system('pip install praw')
import praw
from praw.models import MoreComments
get_ipython().system('pip install textblob')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
get_ipython().system('pip install gensim')
from gensim import matutils, models
import scipy.sparse
from textblob import TextBlob
import pandas as pd
import nltk
import re 
import numpy as np


# In[15]:


#obtaining access to Reddit API
reddit = praw.Reddit(client_id = 'wlWQaMEIZXxrgQ',
                     client_secret = 'sC-GfU6JWXA-ikud7b2G-f_kFZ8',
                     user_agent='BHI Scraper')


# In[16]:


#obtaining current posts
posts = reddit.subreddit('Parenting').hot(limit=3)


# In[17]:


#testing out the posts
for post in posts:
    print(post.selftext, post.author.name, post.created_utc)


# In[ ]:


#creating data table of posts and other information from parenting and daddit
parenting_posts = []
num_posts = 10
r_parenting = reddit.subreddit('Parenting')
r_daddit = reddit.subreddit('daddit')

for post in r_parenting.top(limit = num_posts):
    parenting_posts.append([post.author.name, post.id, post.link_flair_text, post.title, post.created_utc, post.score, post.subreddit, 
                            post.url, post.num_comments, post.selftext, 
                           ' '.join(map(str, [posting.body for posting in list(reddit.redditor(post.author.name).comments.new(limit=None))]))])
    parenting_posts.append([])
    
# for post in r_daddit.hot(limit = num_posts):
#     parenting_posts.append([post.author.name, post.id, post.link_flair_text, post.title,post.created_utc, post.score, post.subreddit, 
#                             post.url, post.num_comments, post.selftext,
#                            ' '.join(map(str, [posting.body for posting in list(reddit.redditor(post.author.name).comments.new(limit=None))]))])
    
parenting_posts = pd.DataFrame(parenting_posts, columns = ['username','id','tag','title','time', 'score', 'subreddit', 'url', 'num_comments', 'text', 'user_text'])
parenting_posts.head()


# In[ ]:


#clearing NaN values and showing resulting table
parenting_posts = parenting_posts[parenting_posts['text'].notna()]
parenting_posts.head()


# In[ ]:


#obtaining the comments from posts and adding it to the table
parenting_comments = []
for ids in parenting_posts['id']:
    parenting_comments.append([comments.body for comments in reddit.submission(id = ids).comments])
parenting_posts['comments'] = parenting_comments
parenting_posts.head()


# In[ ]:


#cleaning comments
def clean_comments(text):
    '''getting rid of the \n in the comments'''
    for comments in text:
        comments = re.sub('\n','', comments)
    return text

def clean_user_text(text):
    text = re.sub('\n','', text)
    return text

cleaning = lambda x: clean_comments(x)
cleaning1 = lambda z: clean_user_text(z)

parenting_posts['comments'] = parenting_posts.comments.apply(cleaning)
parenting_posts['user_text'] = parenting_posts.user_text.apply(cleaning1)
parenting_posts.head()


# In[ ]:


#vectorizing reddit post titles to tag them with topics
cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(parenting_posts.title)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = parenting_posts.id
data_dtm.head()


# In[ ]:


#transposing document-term matrix for topic modeling
tdm = data_dtm.transpose()
tdm.head()


# In[ ]:


#convert to gensim format
sparse_counts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_counts)
id2word = dict((v,k) for k,v in cv.vocabulary_.items())


# In[ ]:


#LDA Topic Modeling for all words
lda = models.LdaModel(corpus=corpus, num_topics=5, id2word=id2word, passes = 80)
lda.print_topics()


# In[ ]:


#LDA Topic Modeling for nouns and adjectives only
from nltk import word_tokenize, pos_tag

def nouns_adj(text):
    '''given the title, tokenize the text and pull out only the nouns'''
    tokenized = word_tokenize(text)
    all_nouns_adj = [word for (word, pos) in pos_tag(tokenized) if pos == 'NN' or pos == 'JJ']
    return ' '.join(all_nouns_adj)

nouns_adj_only = pd.DataFrame(parenting_posts.title.apply(nouns_adj))


# In[ ]:


#vectorize noun_adj words 
cvn = CountVectorizer(stop_words = 'english', max_df=.8)
data_cvn = cvn.fit_transform(nouns_adj_only.title)
data_dtmn = pd.DataFrame(data_cvn.toarray(), columns=cvn.get_feature_names())
data_dtmn.index = parenting_posts.id
data_dtmn.head()


# In[ ]:


# Create the gensim parameteres
corpusn = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmn.transpose()))
id2wordn = dict((v, k) for k, v in cvn.vocabulary_.items())


# In[ ]:


#printing the topics
ldan = models.LdaModel(corpus=corpusn, num_topics=5, id2word=id2wordn, passes=80)
ldan.print_topics()


# In[ ]:


#finding the topics for each post and adding it to the table
topic_dictionary = {0:'family and household', 1:'parenting advice', 
                    2: 'young kids and health', 3:'activities', 4:'routines, kids, and being a parent' }

corpus_transformed = ldan[corpusn]
topics_list = [] 
top_num = []

for topic_breakdown in corpus_transformed:
    prevalent_topic = max(topic_breakdown,key=lambda tup:tup[1])[0]
    top_num.append(prevalent_topic)
    topics_list.append(topic_dictionary[prevalent_topic])
    
parenting_posts['problem_tag'] = topics_list
parenting_posts['topic_num'] = top_num
parenting_posts.head()


# In[ ]:


#obtaining user gender based on text
male_keywords = ['wife','mother', 'mom','she']
female_keywords = ['husband','father','dad','he']

#obtaining child information
girl = ['daughter','girl','daughters',"daugther's","girl's"]
boy = ['son','boy','sons',"son's","boy's"]

def gender_guesser(row):
    '''guess gender based on keywords'''
    text = row['text']
    text = word_tokenize(text)
    text = [word.lower() for word in text]

    gender = 'NaN'
    child_gender = 'NaN'
    
    for word in text:
        if word in girl:
            child_gender = 'F'
        elif word in boy:
            child_gender = 'M'
            
    for word in text:
        if word in male_keywords:
            gender = 'M'
        elif word in female_keywords:
            gender = 'F'
        if row['subreddit'] == 'daddit':
            gender = 'M'
    return { 'gender': gender, 
             'child_gender': child_gender}

guesser = pd.DataFrame(list(parenting_posts.apply(gender_guesser, axis = 1)))
parenting_posts = pd.concat([parenting_posts, guesser], axis = 1)
parenting_posts = parenting_posts[parenting_posts['text'].notna()]
parenting_posts.head()


# In[ ]:


#child age gatherer
numbers = [n for n in range(100)]
numbers = [str(num) for num in numbers]
triggers = ['years', 'Years','yo','y/o', 'weeks','wks']
def child_info(tag):
    ages = 'NaN'
    if tag == None:
        return {'child_age': ages}
    tags = word_tokenize(tag)
    tags = [word.lower() for word in tags]

    for word in tags:
        if word in triggers or word in numbers:
            ages = tag
    return {'child_age': ages}
        

child_information = pd.DataFrame(list(parenting_posts.tag.apply(child_info)))
parenting_posts = pd.concat([parenting_posts, child_information], axis = 1)
parenting_posts = parenting_posts[parenting_posts['text'].notna()]
parenting_posts.head()


# In[ ]:


#cleaning subproblems
from nltk.corpus import wordnet 
sub_problem = ['Hitting','Screaming','Bitting','Crying','Flopping on the floor','Throwing',
               'Waking up early','Feeling tired','Lack of sleep','Resting', 'Eating',
               'Trying','Spitting','Pushing away','Refusing', "Didn't like it",'did not like it',
                'Bitting','Scratching','Throwing','Slapping','Destroying',
               'Pushing','Swearing', 'Avoiding','Getting wet','Being dirty','Fear','Waiting','Sitting']
sub_problem = [word.lower() for word in sub_problem]
#stop_point = 0

#for word in sub_problem:
#    for syn in wordnet.synsets(word):
#        for l in syn.lemmas():
#            sub_problem.append(l.name())
#            stop_point +=1
#            if stop_point >= len(sub_problem) * 5:
#                break

#sub_problem = [word.lower() for word in sub_problem]


# In[ ]:


#obtaining subproblems
def child_problem(tag):
    sub_problems = 'other'
    if tag == None:
        return {'sub_problem': sub_problems}
    tags = word_tokenize(tag)
    tags = [word.lower() for word in tags]

    for word in tags:
        if word in sub_problem:
            sub_problems = word
            
    return {'sub_problem': sub_problems}
        

child_problems = pd.DataFrame(list(parenting_posts.text.apply(child_problem)))
parenting_posts = pd.concat([parenting_posts, child_problems], axis = 1)
parenting_posts = parenting_posts[parenting_posts['text'].notna()]
parenting_posts.head()


# In[ ]:


#final table
parenting_posts.head()


# In[23]:


parenting_posts.to_csv('reddit_parenting_posts1.csv')


# In[24]:


#that's all folks!

