
# coding: utf-8

# In[1]:


import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model


# In[2]:


question = """Install amdgpu drivers on Ubuntu Server 16.04.3 LTS.  I have made it completely through the process of installing the AMDgpu drivers; yet at the end of the process when I run a package query, it finds no matching package.
     wget --referer http://support.amd.com \ > https://www2.ati.com/drivers/linux/ubuntu/amdgpu-pro-17.40-492261.tar.xz

tar xvf amdgpu-pro-17.40-492261.tar.xz

cd /amdgpu-pro-17.40-492261

./amdgpu-pro-install --compute"""


# In[3]:


test_question = [question]
sequences = []


# In[4]:


import pickle

with open('tokernizer.pkl', 'rb') as input:
    tokenizer = pickle.load(input)
    sequences = tokenizer.texts_to_sequences(test_question)


# In[5]:


BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
MAX_SEQUENCE_LENGTH = 200
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.3


# In[6]:


data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


# In[10]:


data


# In[8]:


from keras.models import load_model



model  = load_model("categorize_model_with_embedding.h5")


# In[11]:


pred = model.predict(data)


# In[12]:


pred


# In[ ]:




