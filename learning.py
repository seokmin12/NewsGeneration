import pandas as pd
import numpy as np
from string import punctuation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import os
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

train_data = pd.read_csv('/Users/seokmin/Desktop/project/word_machine_learning/data/news_data.csv')
headline = []

for title in train_data['title'].values:
    title = re.sub(r"[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·a-zA-Z]", "", title)
    headline.append(title)

headline.extend(list(train_data.title.values))
headline = headline[:1000]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(headline)
vocab_size = len(tokenizer.word_index) + 1

sequences = list()

for sentence in headline:
    # 각 샘플에 대한 정수 인코딩
    encoded = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[:i + 1]
        sequences.append(sequence)

index_to_word = {}
for key, value in tokenizer.word_index.items():  # 인덱스를 단어로 바꾸기 위해 index_to_word를 생성
    index_to_word[value] = key

max_len = max(len(l) for l in sequences)

sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')

sequences = np.array(sequences)
X = sequences[:, :-1]
y = sequences[:, -1]

y = to_categorical(y, num_classes=vocab_size)

import pickle

# saving tokenizer
with open('/Users/seokmin/Desktop/project/word_machine_learning/model/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

embedding_dim = 10
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(hidden_units))
model.add(Dense(5000, activation='swish'))
model.add(Dense(vocab_size, activation='softmax'))

mc = ModelCheckpoint('/Users/seokmin/Desktop/project/word_machine_learning/model/news_model.h5', monitor='val_acc',
                     mode='max', verbose=1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, callbacks=[mc])


def sentence_generation(model, tokenizer, current_word, n):  # 모델, 토크나이저, 현재 단어, 반복할 횟수
    init_word = current_word
    sentence = ''

    # n번 반복
    for _ in range(n):
        encoded = tokenizer.texts_to_sequences([current_word])[0]
        encoded = pad_sequences([encoded], maxlen=max_len - 1, padding='pre')

        # 입력한 X(현재 단어)에 대해서 y를 예측하고 y(예측한 단어)를 result에 저장.
        result = model.predict(encoded, verbose=0)
        result = np.argmax(result, axis=1)

        for word, index in tokenizer.word_index.items():
            # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면
            if index == result:
                break

        # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
        current_word = current_word + ' ' + word

        # 예측 단어를 문장에 저장
        sentence = sentence + ' ' + word

    sentence = init_word + sentence
    return sentence


print(sentence_generation(model, tokenizer, '손흥민', 10))
