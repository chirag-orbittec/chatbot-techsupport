
# coding: utf-8

# In[ ]:





# In[1]:


import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv("categories_data.csv",encoding='utf-8')


# In[3]:


train, test = train_test_split(df, test_size=0.2)


# In[ ]:


train.head()


# In[ ]:


df.shape


# In[4]:


dataset_train = train.values
dataset_test = test.values



# In[5]:


questions_train = dataset_train[:,0:1]
questions_test = dataset_test[:,0:1]

# In[6]:


def stringToList(string):
    tempList = string.split()
    newList = list(map(lambda x: int(x), tempList))
    return(newList)


# In[7]:



train_questions_int = []

for item in questions_train:
    intlist=stringToList(str(item[0]))
    intlist=intlist+([0]*(160-len(intlist)))
    train_questions_int.append(intlist)
    


# In[8]:


test_questions_int = []

for item in questions_test:
    intlist=stringToList(str(item[0]))
    intlist=intlist+([0]*(160-len(intlist)))
    test_questions_int.append(intlist)


# In[10]:


import numpy as np

train_questions_int = np.asarray(train_questions_int,dtype=np.float32)
test_questions_int = np.asarray(train_questions_int,dtype=np.float32)


# In[11]:


train_labels = dataset_train[:,1:]
test_labels = dataset_test[:,1:]


# In[12]:


def baseline_model():
    model = Sequential()
    model.add(Dense(160, input_dim=160, init='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(150, init='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(150, init='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(150, init='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(132, init='normal', activation='sigmoid'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


# In[ ]:


#model.fit(train_questions_int, train_labels, epochs=1, batch_size=2000)


# In[ ]:

seed = 7
np.random.seed(seed)
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=2000)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, train_questions_int, train_labels, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



