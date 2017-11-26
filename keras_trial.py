
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


BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
MAX_SEQUENCE_LENGTH = 200
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.3


# In[3]:


print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


# In[4]:


import pandas as pd


df = pd.read_csv("../categories_data_raw.csv")


# In[5]:


questions_train = df.values[:,0:1]


# In[6]:


questions_train


# In[7]:


texts = []



for item in questions_train:
    strlist=str(item[0])
    texts.append(strlist)



# In[8]:


tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)


# In[9]:


word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[10]:


data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


# In[11]:


train_labels = df.values[:,1:]


# In[12]:


data.shape


# In[13]:


train_labels.shape


# In[14]:


# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = train_labels[indices]
TEST_SPLIT = 0.10
VAL_SPLIT = 0.20
num_test_samples = int(TEST_SPLIT * data.shape[0])
num_validation_samples = int(VAL_SPLIT * data.shape[0])
x_test = data[:num_test_samples]
y_test = labels[:num_test_samples]
x_val = data[num_test_samples:num_test_samples+num_validation_samples]
y_val = labels[num_test_samples:num_test_samples+num_validation_samples]
x_train = data[num_test_samples+num_validation_samples:]
y_train = labels[num_test_samples+num_validation_samples:]


# In[16]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_val.shape)
print(y_val.shape)


# In[24]:


print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# In[26]:


# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(36, activation='softmax')(x)


# In[ ]:


model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val))


# In[ ]:


preds = model.predict(x_test)


# In[ ]:


print(preds)

