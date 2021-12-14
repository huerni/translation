import pandas as pd
import os
import re
import string
from zhon.hanzi import punctuation
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class Datasets(object):
	def __init__(self, english_path='tranlation_data/en20180913s.txt', chinese_path='tranlation_data/zhtoken20180913s.txt'):

		self.input_path = english_path
		self.target_path = chinese_path


	def getData(self):
		'''按行转化为列表'''
		input_texts = []
		target_texts = []
		with open(self.input_path, 'r', encoding='utf-8') as f:
			lines = f.read().split('\n')
			for line in lines:
				input_texts.append(line)

		with open(self.target_path, 'r', encoding='utf-8') as f:
			lines = f.read().split('\n')
			for line in lines:
				target_texts.append(line)

		'''转化为df'''
		df = pd.DataFrame({'English': input_texts, 'Chinese': target_texts})
		df = df.iloc[:-1, :]

		'''对英文处理掉无效单词'''
		df['English'] = df['English'].apply(self.preprocess_sentence)

		'''删除中英文标点符号，并将中文前后加上起止符 sos和eos'''
		translator = str.maketrans('', '', string.punctuation)
		df.English = df.English.apply(lambda x: x.translate(translator))
		translator = str.maketrans('', '', punctuation)
		df.Chinese = df.Chinese.apply(lambda x: x.translate(translator))
		df.Chinese = df.Chinese.apply(lambda x: 'sos ' + x + ' eos')

		eng_texts = df.English.to_list()
		chi_texts = df.Chinese.to_list()

		'''转化为tokenizer'''
		eng_tokenizer, eng_encoded = self.tokenize_sent(text=eng_texts)
		chi_tokenizer, chi_encoded = self.tokenize_sent(text=chi_texts)

		eng_index_word = eng_tokenizer.index_word  # 字符与数字相对应的dict
		chi_index_word = chi_tokenizer.index_word
		chi_word_index = chi_tokenizer.word_index

		'''词典长度'''
		CHI_VOCAB_SIZE = len(chi_tokenizer.word_counts) + 1
		ENG_VOCAB_SIZE = len(eng_tokenizer.word_counts) + 1

		max_eng_len = 0
		for i in range(len(eng_encoded)):
			if len(eng_encoded[i]) > max_eng_len:
				max_eng_len = len(eng_encoded[i])

		max_chi_len = 0
		for i in range(len(chi_encoded)):
			if len(chi_encoded[i]) > max_chi_len:
				max_chi_len = len(chi_encoded[i])

		'''将中英文向量分别补成同样长度'''
		eng_padded = pad_sequences(eng_encoded, maxlen=max_eng_len, padding='post')
		chi_padded = pad_sequences(chi_encoded, maxlen=max_chi_len, padding='post')
		eng_padded = np.array(eng_padded)
		chi_padded = np.array(chi_padded)

		'''划分训练集，测试集'''
		x_train, x_test, y_train, y_test = train_test_split(eng_padded, chi_padded, test_size=0.1, random_state=0)

		return x_train, x_test, y_train, y_test

	def preprocess_sentence(self, sentence):
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

	def tokenize_sent(self, text):
		tokenizer = Tokenizer()  # 文本标记
		tokenizer.fit_on_texts(text)  # 学习出文本的字典
		return tokenizer, tokenizer.texts_to_sequences(text)  # 每个词转成数字