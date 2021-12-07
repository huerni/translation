import tensorflow as tf
from tensorflow import keras
import pandas as pd
import os
import re
import string
from zhon.hanzi import punctuation
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import LSTM, Dropout, Dense, Embedding, Bidirectional, Add, Concatenate, Dropout
from tensorflow.keras import Input, Model
from tensorflow.keras.utils import plot_model

#pd.set_option('max_colwidth', 500)

def preprocess_sentence(sentence):
  sentence = sentence.lower().strip()
  # removing contractions
  sentence = re.sub(r"i'm", "i am", sentence)
  sentence = re.sub(r"he's", "he is", sentence)
  sentence = re.sub(r"she's", "she is", sentence)
  sentence = re.sub(r"it's", "it is", sentence)
  sentence = re.sub(r"that's", "that is", sentence)
  sentence = re.sub(r"what's", "that is", sentence)
  sentence = re.sub(r"where's", "where is", sentence)
  sentence = re.sub(r"how's", "how is", sentence)
  sentence = re.sub(r"\'ll", " will", sentence)
  sentence = re.sub(r"\'ve", " have", sentence)
  sentence = re.sub(r"\'re", " are", sentence)
  sentence = re.sub(r"\'d", " would", sentence)
  sentence = re.sub(r"\'re", " are", sentence)
  sentence = re.sub(r"won't", "will not", sentence)
  sentence = re.sub(r"can't", "cannot", sentence)
  sentence = re.sub(r"n't", " not", sentence)
  sentence = re.sub(r"n'", "ng", sentence)
  sentence = re.sub(r"'bout", "about", sentence)
  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
  sentence = sentence.strip()
  return sentence

def tokenize_sent(text):
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(text)
  return tokenizer, tokenizer.texts_to_sequences(text)

input_path = 'tranlation_data/en20180913s.txt'
target_path = 'tranlation_data/zhtoken20180913s.txt'

input_texts = []
target_texts = []
with open(input_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
    for line in lines:
        input_texts.append(line)

with open(target_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
    for line in lines:
        target_texts.append(line)

df = pd.DataFrame({'English':input_texts, 'Chinese':target_texts})
df = df.iloc[:-1, :]

df['English'] = df['English'].apply(preprocess_sentence)

translator = str.maketrans('', '', string.punctuation)
df.English= df.English.apply(lambda x: x.translate(translator))
translator = str.maketrans('', '', punctuation)
df.Chinese= df.Chinese.apply(lambda x: x.translate(translator))
df.Chinese = df.Chinese.apply(lambda x: 'sos '+ x +' eos')

eng_texts = df.English.to_list()
chi_texts = df.Chinese.to_list()

eng_tokenizer, eng_encoded= tokenize_sent(text= eng_texts)
chi_tokenizer, chi_encoded= tokenize_sent(text= chi_texts)

eng_index_word = eng_tokenizer.index_word
chi_index_word= chi_tokenizer.index_word
chi_word_index = chi_tokenizer.word_index

CHI_VOCAB_SIZE=len(chi_tokenizer.word_counts)+1
ENG_VOCAB_SIZE = len(eng_tokenizer.word_counts)+1

max_eng_len = 0
for i in range(len(eng_encoded)):
  if len(eng_encoded[i]) > max_eng_len:
    max_eng_len= len(eng_encoded[i])

max_chi_len = 0
for i in range(len(chi_encoded)):
  if len(chi_encoded[i]) > max_chi_len:
    max_chi_len = len(chi_encoded[i])

eng_padded = pad_sequences(eng_encoded, maxlen=max_eng_len, padding='post')
chi_padded = pad_sequences(chi_encoded, maxlen=max_chi_len, padding='post')
eng_padded= np.array(eng_padded)
ara_padded= np.array(chi_padded)

x_train, x_test, y_train, y_test = train_test_split(eng_padded, chi_padded, test_size=0.1, random_state=0)

# Encoder
encoder_input = Input(shape=(None, ))
encoder_embd = Embedding(ENG_VOCAB_SIZE,1024, mask_zero=True)(encoder_input)
encoder_lstm = Bidirectional(LSTM(512, return_state=True))
encoder_output, forw_state_h, forw_state_c, back_state_h, back_state_c = encoder_lstm(encoder_embd)
state_h_final = Concatenate()([forw_state_h, back_state_h])
state_c_final = Concatenate()([forw_state_c, back_state_c])

# Now take only states and create context vector
encoder_states= [state_h_final, state_c_final]

# Decoder
decoder_input = Input(shape=(None,))
# For zero padding we have added +1 in arabic vocab size
decoder_embd = Embedding(CHI_VOCAB_SIZE, 1024, mask_zero=True)
decoder_embedding= decoder_embd(decoder_input)
# We used bidirectional layer above so we have to double units of this lstm
decoder_lstm = LSTM(1024, return_state=True,return_sequences=True )
# just take output of this decoder dont need self states
decoder_outputs, _, _= decoder_lstm(decoder_embedding, initial_state=encoder_states)
# here this is going to predicct so we can add dense layer here
# here we want to convert predicted numbers into probability so use softmax
decoder_dense= Dense(CHI_VOCAB_SIZE, activation='softmax')
# We will again feed predicted output into decoder to predict its next word
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_input, decoder_input], decoder_outputs)
#plot_model(model)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
EPOCHS= 100
# Training
encoder_input_data = x_train
decoder_input_data = y_train[:,:-1]
decoder_target_data = y_train[:,1:]

# Testing
encoder_input_test = x_test
decoder_input_test = y_test[:,:-1]
decoder_target_test= y_test[:,1:]

history = model.fit([encoder_input_data, decoder_input_data],decoder_target_data,
                    epochs=EPOCHS,
                    batch_size=128,
                    validation_data = ([encoder_input_test, decoder_input_test],decoder_target_test ))