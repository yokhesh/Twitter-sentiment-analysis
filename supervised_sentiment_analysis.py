import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import nltk
from nltk.corpus import stopwords
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

data= pd.read_csv("result.csv")
data = data.drop(["Unnamed: 0"], axis=1)

data = data.sample(frac=1).reset_index(drop=True)
text = data['blog_text']
label = data['sentiment']

onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
label  = np.array(label)
label  = label .reshape(len(label ), 1)
label  = onehot_encoder.fit_transform(label )

'''
onehot_encoder = OneHotEncoder(sparse=False,categories='auto')

train = data.iloc[:750]
train_data = train['blog_text']
train_target = train['sentiment']
train_target = np.array(train_target)
train_target = train_target.reshape(len(train_target), 1)
train_target = onehot_encoder.fit_transform(train_target)

test = data.iloc[751:]
test_data = test['blog_text']
test_target = test['sentiment']
test_target = np.array(test_target)
test_target = test_target.reshape(len(test_target), 1)
test_target = onehot_encoder.fit_transform(test_target)

'''
X_train, X_val_test, y_train, y_val_test = train_test_split(text, label, test_size=0.25, random_state=1000)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.25, random_state=1000)
'''
###vectorizer
vectorizer = CountVectorizer()
vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train)
X_test  = vectorizer.transform(X_test)
'''
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)

'''
##########logistic regression
classifier = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=300)
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
'''
def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix
embedding_dim = 50
embedding_matrix = create_embedding_matrix('D:/Yokhesh/semester2/data_analytics/project/glove.6B/glove.6B.50d.txt',tokenizer.word_index, embedding_dim)

#####Basic Neural Network#####

model = Sequential()

'''
model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim,
                           weights=[embedding_matrix], 
                           input_length=maxlen, 
                           trainable=True))
'''
model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim,
                           input_length=maxlen))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPool1D())
#model.add(layers.Flatten())
model.add(layers.Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(layers.Dense(16, input_dim=X_train.shape[1], activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train,epochs=20,verbose=2,validation_data=(X_val, y_val))

'''
def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPool1D())
#model.add(layers.Flatten())
    model.add(layers.Dense(10, input_dim=X_train.shape[1], activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

param_grid = dict(num_filters=[32, 64, 128],
                  kernel_size=[3, 5, 7],
                  vocab_size=[5000], 
                  embedding_dim=[50],
                  maxlen=[100])
model = KerasClassifier(build_fn=create_model,
                            epochs=20, batch_size=100,
                            verbose=2)
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                              cv=4, verbose=1, n_iter=5)
grid_result = grid.fit(X_train, y_train)
'''
def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
