import tensorflow as tf
from keras.layers import Reshape
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

from tensorflow.keras.layers import LSTM, Dropout, Dense, Embedding, Bidirectional, Add, Concatenate, Dropout, Activation, BatchNormalization, TimeDistributed, \
    RepeatVector, Dot
from tensorflow.keras import Input, Model
from util import preprocess_sentence


def tokenize_sent(text):
  tokenizer = Tokenizer()  # 文本标记
  tokenizer.fit_on_texts(text) # 学习出文本的字典
  return tokenizer, tokenizer.texts_to_sequences(text) #　每个词转成数字


'''读取中英文数据'''
input_path = 'tranlation_data/en20180913s.txt'
target_path = 'tranlation_data/zhtoken20180913s.txt'

'''按行转化为列表'''
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

'''转化为df'''
df = pd.DataFrame({'English': input_texts, 'Chinese': target_texts})
df = df.iloc[:-1, :]

'''对英文处理掉无效单词'''
df['English'] = df['English'].apply(preprocess_sentence)

'''删除中英文标点符号，并将中文前后加上起止符 sos和eos'''
translator = str.maketrans('', '', string.punctuation)
df.English = df.English.apply(lambda x: x.translate(translator))
translator = str.maketrans('', '', punctuation)
df.Chinese = df.Chinese.apply(lambda x: x.translate(translator))
df.Chinese = df.Chinese.apply(lambda x: 'sos ' + x + ' eos')

eng_texts = df.English.to_list()
chi_texts = df.Chinese.to_list()

'''转化为tokenizer'''
eng_tokenizer, eng_encoded= tokenize_sent(text = eng_texts)
chi_tokenizer, chi_encoded= tokenize_sent(text = chi_texts)

eng_index_word = eng_tokenizer.index_word  # 字符与数字相对应的dict
chi_index_word= chi_tokenizer.index_word
chi_word_index = chi_tokenizer.word_index

'''词典长度'''
CHI_VOCAB_SIZE = len(chi_tokenizer.word_counts)+1
ENG_VOCAB_SIZE = len(eng_tokenizer.word_counts)+1

max_eng_len = 0
for i in range(len(eng_encoded)):
    if len(eng_encoded[i]) > max_eng_len:
        max_eng_len= len(eng_encoded[i])

max_chi_len = 0
for i in range(len(chi_encoded)):
    if len(chi_encoded[i]) > max_chi_len:
        max_chi_len = len(chi_encoded[i])

'''将中英文向量分别补成同样长度'''
eng_padded = pad_sequences(eng_encoded, maxlen=max_eng_len, padding='post')
chi_padded = pad_sequences(chi_encoded, maxlen=max_chi_len, padding='post')
eng_padded= np.array(eng_padded)
chi_padded= np.array(chi_padded)

'''划分训练集，测试集'''
x_train, x_test, y_train, y_test = train_test_split(eng_padded, chi_padded, test_size=0.1, random_state=0)



def one_step_attention(a, s_prev):
    """
    Attention机制的实现，返回加权后的Context Vector

    @param a: BiRNN的隐层状态
    @param s_prev: Decoder端LSTM的上一轮隐层输出

    Returns:
    context: 加权后的Context Vector
    """

    # 将s_prev复制max_eng_len次
    repeator = RepeatVector(max_eng_len)
    concatenator = Concatenate(axis=-1)
    densor_tanh = Dense(32, activation="tanh")
    densor_relu = Dense(1, activation="relu")
    activator = Activation('softmax', name='attention_weights')
    dotor = Dot(axes=1)
    s_prev = repeator(s_prev)
    # 拼接BiRNN隐层状态与s_prev
    concat = concatenator([a, s_prev])
    # 计算energies
    e = densor_tanh(concat)
    energies = densor_relu(e)
    # 计算weights
    alphas = activator(energies)
    # 加权得到Context Vector
    context = dotor([alphas, a])

    return context

'''编码器Encoder'''
encoder_input = Input(shape=(None, )) # 输入层
encoder_embd = Embedding(ENG_VOCAB_SIZE, 256, mask_zero=True)(encoder_input) # 嵌入层
encoder_lstm = Bidirectional(LSTM(128, return_state=True, return_sequences=True)) # BiLSTM层
'''输入嵌入层输出，输出 编码输出，前向状态  后向状态'''
encoder_output, forw_state_h, forw_state_c, back_state_h, back_state_c = encoder_lstm(encoder_embd)
state_h_final = Concatenate()([forw_state_h, back_state_h])
state_c_final = Concatenate()([forw_state_c, back_state_c])

'''上下文向量'''
encoder_states = [state_h_final, state_c_final]

'''解码器Decoder'''
decoder_input = Input(shape=(None,)) # 输入层
decoder_embd = Embedding(CHI_VOCAB_SIZE, 256, mask_zero=True) # 嵌入层
decoder_embedding= decoder_embd(decoder_input)
decoder_lstm = LSTM(256, return_state=True, return_sequences=True ) # LSTM层
reshapor = Reshape((1, len(CHI_VOCAB_SIZE)))
concator = Concatenate(axis=-1)
out0 = Input(shape=(CHI_VOCAB_SIZE, ), name='out0')
out = reshapor(out0)
for t in range(max_chi_len):
    context = one_step_attention(encoder_output, encoder_states)
    context = concator([context, reshapor(out)])

decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states) # 解码器输出
decoder_dense = Dense(CHI_VOCAB_SIZE, activation='softmax') # 重建层
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_input, decoder_input], decoder_outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
EPOCHS = 100

# Training
'''训练数据'''
encoder_input_data = x_train
decoder_input_data = y_train[:, :-1]
decoder_target_data = y_train[:, 1:]

# Testing
'''测试数据'''
encoder_input_test = x_test
decoder_input_test = y_test[:, :-1]
decoder_target_test= y_test[:, 1:]

history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                    epochs=EPOCHS,
                    batch_size=128,
                    validation_data=([encoder_input_test, decoder_input_test], decoder_target_test))



