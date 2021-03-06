import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import LSTM, Dropout, Dense, Embedding, Bidirectional, Add, Concatenate, Dropout
from tensorflow.keras import Input, Model
import pickle

with open('eng_tokenizer.pickle', 'rb') as handle:
	eng_tokenizer = pickle.load(handle)
with open('chi_tokenizer.pickle', 'rb') as handle:
	chi_tokenizer = pickle.load(handle)

eng_index_word = eng_tokenizer.index_word  # 字符与数字相对应的dict
chi_index_word= chi_tokenizer.index_word
chi_word_index = chi_tokenizer.word_index

'''词典长度'''
CHI_VOCAB_SIZE = len(chi_tokenizer.word_counts)+1
ENG_VOCAB_SIZE = len(eng_tokenizer.word_counts)+1


def get_predicted_sentence(input_seq):
	# 对输入编码， 得到状态向量
	states_value = encoder_model.predict(input_seq)

	# 空的长度为1的目标序列
	target_seq = np.zeros((1, 1))

	# 用开始字符填充目标序列的第一个字符。
	target_seq[0, 0] = chi_word_index['sos']

	stop_condition = False
	decoded_sentence = ''

	while not stop_condition:
		output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
		sampled_token_index = np.argmax(output_tokens[0, -1, :])
		if sampled_token_index == 0:
			break
		else:
			sampled_char = chi_index_word[sampled_token_index]

		decoded_sentence += ' ' + sampled_char

		if sampled_char == 'eos':
			stop_condition = True

		target_seq = np.zeros((1, 1))
		target_seq[0, 0] = sampled_token_index

		states_value = [h, c]

	return decoded_sentence


def get_chinese_sentence(sequence):
	sentence = ""
	for i in sequence:
		if ((i != 0 and i != chi_word_index['sos']) and i != chi_word_index['eos']):
			sentence = sentence + chi_index_word[i] + ' '
	return sentence


def get_eng_sent(sequence):
	sentence = ''
	for i in sequence:
		if (i != 0):
			sentence = sentence + eng_index_word[i] + ' '
	return sentence


'''读取模型'''
model = tf.keras.models.load_model('./modellstm (1).h5')
encoder_output, forw_state_h, forw_state_c, back_state_h, back_state_c = model.get_layer('bidirectional_1').output
state_h_final = Concatenate()([forw_state_h, back_state_h])
state_c_final = Concatenate()([forw_state_c, back_state_c])
encoder_states = [state_h_final, state_c_final]

encoder_model = Model(model.get_layer('input_3').input, encoder_states)

decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_input = [decoder_state_input_h, decoder_state_input_c]
decoder_embd = model.get_layer('embedding_3')
decoder_input = model.get_layer('embedding_3').input
dec_embd2 = decoder_embd(decoder_input)
decoder_lstm = model.get_layer('lstm_3')
decoder_output2, state_h2, state_c2 = decoder_lstm(dec_embd2, initial_state=decoder_states_input)
deccoder_states2 = [state_h2, state_c2]
decoder_dense = model.get_layer('dense')
decoder_output2 = decoder_dense(decoder_output2)

decoder_model = Model([decoder_input] + decoder_states_input, [decoder_output2] + deccoder_states2)

#for i in range(1):
#	print(x_test[i].shape)
#	print("English sentence:", get_eng_sent(x_test[i]))
#	print("Actual Chinese Sentence:", get_chinese_sentence(y_test[i]))
#	print("Translated Chinese Sentence:", get_predicted_sentence(x_test[i].reshape(1, 109))[:-4])
#	print("\n")
