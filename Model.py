import matplotlib
import matplotlib.pyplot as plt
import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
from keras import layers
import itertools
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
df = pd.read_csv('MedicalRecord.csv')
df = df[pd.notnull(df['c_group'])]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import Callback, ModelCheckpoint
from keras import utils
from keras.optimizers import SGD
from keras.layers import Embedding
from keras.layers import LSTM,Conv1D,GlobalAveragePooling1D,Flatten
from keras.metrics import top_k_categorical_accuracy

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

df['full_text'] = df[['full_text','diag_name']].apply(lambda x: ''.join(x), axis=1)
df['full_text'] = df['full_text'].apply(clean_text)
train_size = int(len(df))

#train_posts = df['diag_name'].apply(clean_text)

train_posts = df['full_text']
max_words = len(df.full_text.unique())
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_posts) # only fit on train
x_train = tokenize.texts_to_matrix(train_posts,mode="tfidf")
batch_size = 50
epochs = 10
kf = KFold(n_splits=10, random_state=None, shuffle=True)
def top_7_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=7)
#for train_index, test_index in kf.split(x_train):
# Build the model
d = {}
arr = []
# for (i,v) in enumerate(df.c_group.value_counts()):
#     #print(df.c_group.value_counts().keys()[i],v)
#     if (v == 1):
#         arr.append(df.c_group.value_counts().keys()[i])
#

# for (i,v) in enumerate(df.c_group):
#     df.c_group[i] = df.c_group[i][:1]
#     d[df.c_group[i]] = d.get(df.c_group[i], 0) + 1


arr = ["L","I","E","K","R","B","J","N","F","H"]
for (i,v) in enumerate(df.c_group):
    if len(list(filter(lambda x:x == v[:1],arr))) > 0:
        df.c_group[i] = "O"


#     else:
#         df.c_group[i] = df.c_group[i][:1]


train_tags = df['c_group']
encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)


X_train = x_train
Y_train = y_train
#X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.1)

#train word vector
from keras.regularizers import l1
arr = list(map(lambda x:x[:1],df.c_group))
encoder = LabelEncoder()
encoder.fit(arr)
arr = encoder.transform(arr)
class_len = np.max(arr) + 1
arr = utils.to_categorical(arr, class_len)
from sklearn.model_selection import KFold
from numpy import zeros
from numpy import asarray
max_length = 0
for i in X_train:
    if max_length < len(i):
        max_length = len(i)
vocab_size = len(tokenize.word_index) + 1
kf = KFold(n_splits=10, random_state=None, shuffle=True)
word_index = tokenize.word_index
MAX_NUM_WORDS = 35155
EMBEDDING_DIM = 100
embeddings_index = {}
# can be any word embedding weight example = enwiki_20180420_100d.txt
with open(os.path.join("", 'enwiki_20180420_100d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except ValueError:
            pass

print('Found %s word vectors.' % len(embeddings_index))
# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
print(embedding_matrix)
#embedding_matrix = np.matmul(embeddings,embedding_matrix.T)

# import random
# from numpy  import array
# print(random.uniform(-1,1))
# embedding_matrix = list()
# for i in range(0,35156):
#     embedding_matrix.append([])
#     for j in range(0,100):
#         embedding_matrix[i].append(random.uniform(-1,1))
# print("Random Finished")
# embedding_matrix = array(embedding_matrix)


from keras.layers import GRU
from keras.regularizers import l1
from sklearn.metrics import classification_report


with tf.device('/gpu:0'):
    for train_index, test_index in kf.split(X_train):
        model = Sequential()
        model.add(Embedding(vocab_size, 100, input_length=max_length,weights=[embedding_matrix],trainable=False))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(int(num_classes/8),activation="tanh",activity_regularizer=l1(0.001), input_shape=(max_words,)))
        model.add(layers.Dropout(0.5))
        model.add(Dense(int(num_classes/1),activation="tanh"))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes,activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer="adam",
                      metrics=['accuracy'])
        checkpoint = ModelCheckpoint('best_model_word2vec.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
        # with tf.device('/gpu:0'):
        model.summary()
        history = model.fit(X_train[train_index], Y_train[train_index],
                        batch_size=batch_size,
                        epochs=50,
                        verbose=1,
                        validation_split=0.1,callbacks=[checkpoint])
        score = model.evaluate(X_train[test_index], Y_train[test_index],
                           batch_size=batch_size)
        print('Test accuracy:', score)
        Y_pred = model.predict_classes(X_train[test_index])
        test = utils.to_categorical(Y_pred, num_classes)
        report = classification_report(Y_train[test_index], test,target_names=sorted(df.c_group.unique()))
        print(report)
        break
print(history.history['acc'])
print(history.history['val_acc'])
print(history.history['loss'])
print(history.history['val_loss'])
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
