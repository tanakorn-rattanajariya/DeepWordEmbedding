import matplotlib
import matplotlib.pyplot as plt
import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
import itertools
import os

df = pd.read_csv('MedicalRecord.csv')
df = df[pd.notnull(df['c_group'])]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import Callback, ModelCheckpoint
from keras import utils
from keras.engine.topology import Layer
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text
df['full_text'] = df['full_text'].apply(clean_text)
df['full_text'] = df[['full_text','diag_name']].apply(lambda x: ''.join(x), axis=1)
df = df[pd.notnull(df['c_group'])]
df['diag'] = df[['c_group','diag_name']].apply(lambda x: ':'.join(x), axis=1)
train_size = int(len(df))
train_posts = df['full_text'][:train_size]
max_words = len(df.full_text.unique())
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_posts) # only fit on train
min = int(0*len(df['full_text']))
max = int(1*len(df['full_text']))
#train_posts = df['full_text'][0]
train_posts = ["feeling of pain at knee"]
x_train = tokenize.texts_to_matrix(train_posts,mode="tfidf")
x_test = x_train[:]

from keras.models import load_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
arr = []
arr = ["L","I","E","K","R","B","J","N","F","H"]
for (i,v) in enumerate(df.c_group):
    if len(list(filter(lambda x:x == v[:1],arr))) > 0:
        df.c_group[i] = "O"
model = load_model('best_model_word2vec.h5')
y_pred = model.predict_classes(x_test)
#print(y_pred)
from heapq import nlargest
y_pred = list(map(lambda x: sorted(df.c_group.unique())[x], y_pred))
sum = 0
index = 0
result = []
for (i,v) in enumerate(model.predict_proba(x_test)):
    arr = []
    for j in nlargest(7, enumerate(v),key=lambda x: x[1]):
        arr.append(j)
    result.append(arr)
#print(result)
y_actual = df.c_group[min:max]
x = []
y = []
for (i,v) in enumerate(y_pred):
    for j in result[0]:
        if (j[0] == 124):
            print("Other diseases")
        else:
            print(df[df['c_group'].str.match(sorted(df.c_group.unique())[j[0]])]['diag_name'].tolist()[0])
        print(sorted(df.c_group.unique())[j[0]],str(round(j[1]*100,2))+"%")
        x.append(sorted(df.c_group.unique())[j[0]])
        y.append(round(j[1]*100,2))
#      if sorted(df.c_group.unique())[v] == y_train[i]:
#         print(sorted(df.c_group.unique())[j[0]],round(j[1]*100,2))
#         acc += 1
#         print(sorted(df.c_group.unique())[j[0]],round(j[1]*100,2))
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# f, ax = plt.subplots(1,figsize=(20,10))
# n = np.arange(7)
# lineWidth = 2
# ax.plot(n,y,linewidth=lineWidth)
# ax.set_ylim(bottom=0)
# plt.show(f)
print(y)
print(y)
s = 0
for v in y:
    s += v
y.append(100-s)
x.append("Not in top 7th list")
print(y)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
f, ax = plt.subplots(1,figsize=(20,10))
lineWidth = 2
n = np.arange(len(y))
ax.plot(x,y,linewidth=lineWidth)
ax.set_ylim(bottom=0)
plt.show(f)
