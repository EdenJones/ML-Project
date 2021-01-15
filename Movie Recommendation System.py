#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import warnings


# In[4]:


warnings.filterwarnings('ignore')


# In[59]:


columns_names=['user_id','item_id','rating','timestamp']
df=pd.read_csv('u.data',sep="\t",names=columns_name)


# In[60]:


df.head()


# In[14]:


df.shape


# In[73]:


df['user_id']


# In[64]:


df['user_id'].nunique()


# In[65]:


df['item_id'].nunique()


# In[67]:


movies_title=pd.read_csv('u.item',sep="\|",header=None)


# In[68]:


movies_title.shape


# In[70]:


movies_title.head()


# In[92]:


df.tail()#Gives the last five entries


# In[88]:


df=pd.merge(df,movies_title,on="item_id")


# In[89]:


df


# In[93]:


ratings=pd.DataFrame(df.groupby('title').mean()['rating'])


# In[94]:


ratings.head()


# In[95]:


ratings['num of ratings']=pd.DataFrame(df.groupby('title').count()['rating'])


# #Creating the recommendation System 

# In[96]:


df.head()


# In[97]:


moviemat=df.pivot_table(index="user_id",columns="title",values="rating")


# In[98]:


moviemat.head()


# In[99]:


starwars_user_ratings=moviemat['Star Wars (1977)']


# In[101]:


starwars_user_ratings.head()


# In[103]:


starwars_user_ratings.head(20)


# In[104]:


similar_to_starwars=moviemat.corrwith(starwars_user_ratings)


# In[105]:


similar_to_starwars


# In[107]:


corr_starwars=pd.DataFrame(similar_to_starwars,columns=['correlation'])


# In[108]:


corr_starwars.dropna(inplace=True)


# In[109]:


corr_starwars


# In[110]:


corr_starwars.head()


# In[112]:


corr_starwars.sort_values('correlation',ascending=False).head(10)


# In[113]:


corr_starwars=corr_starwars.join(ratings['num of ratings'])


# In[114]:


corr_starwars


# In[115]:


corr_starwars.head()


# In[119]:


corr_starwars[corr_starwars['num of ratings']>100].sort_values('correlation',ascending=False)


# In[141]:


def predict_movies(movie_name):
    movie_user_ratings=moviemat[movie_name]
    similar_to_movie=moviemat.corrwith(movie_user_ratings)
    corr_movie=pd.DataFrame(similar_to_movie,columns=['correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie=corr_movie.join(ratings['num of ratings'])
    
    predictions=corr_movie[corr_movie['num of ratings']>100].sort_values('correlation',ascending=False)
    
    return predictions


# In[145]:


predict_my_movie=predict_movies("Titanic (1997)")


# In[146]:


predict_my_movie.head()


# In[ ]:




