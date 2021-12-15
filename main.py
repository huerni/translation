import matplotlib
matplotlib.use('TkAgg')
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import networkx as nx
import random
from tkinter import ttk
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dropout, Dense, Embedding, Bidirectional, Add, Concatenate, Dropout
from tensorflow.keras import Input, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from util import preprocess_sentence
import string
import pickle
from predict_model import get_predicted_sentence
import numpy as np

root = tkinter.Tk()  # 创建tkinter的主窗口
root.tk.call("source", "azure.tcl")
root.tk.call("set_theme", "light")
root.wm_title("Translation")
root.iconbitmap('./logo.ico')
w = root.winfo_screenwidth()
h = root.winfo_screenheight()
root.geometry("%dx%d+%d+%d" % (1000, 600, (w/2)-500, (h/2)-300))
root.configure(bg='white')

fm1 = ttk.Frame(root)
fm1.pack(side=tkinter.TOP,  # 上对齐
		fill=tkinter.BOTH,  # 填充方式
		expand=1)

fm12 = ttk.Frame(root)
fm12.pack(side=tkinter.TOP)
fm2 = ttk.Frame(root)
fm2.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


with open('eng_tokenizer.pickle', 'rb') as handle:
	eng_tokenizer = pickle.load(handle)


def start_translation(event, *args):
	chinese_entry.configure(state="normal")
	english_pre_value = english_entry.get("0.0", "end")
	english_list = english_pre_value.split('.')
	chinese_entry.delete("0.0", "end")
	for e in english_list:
		english_value = e.strip()
		if english_value == "": continue
		print(english_value)
		translator = str.maketrans('', '', string.punctuation)
		english_value = english_value.translate(translator)
		english_value = [str(english_value)]
		english_value = eng_tokenizer.texts_to_sequences(english_value)
		eng_padded = pad_sequences(english_value, maxlen=109, padding='post')
		eng_padded = np.array(eng_padded)
		chinese_entry.insert("end", get_predicted_sentence(eng_padded[0].reshape(1, 109))[:-4] + "。")
	chinese_entry.configure(state="disabled")
	return 'break'

def button_delete(event, *args):
	if english_entry.get("0.0", "end").strip() == "ENTER键开始翻译".strip():
		english_entry.delete("0.0", "end")

e_label=ttk.Label(fm1, text="英语: ", font=('Calibri', 15))
e_label.pack(side=tkinter.LEFT)
'''英语输入'''
scrollbar_v = tkinter.Scrollbar(fm1)
scrollbar_v.pack(side=tkinter.RIGHT, fill=tkinter.Y)
english_entry = tkinter.Text(fm1, font=("Calibri 14"), width=100, height=13,  yscrollcommand=scrollbar_v.set, wrap=tkinter.CHAR, highlightcolor='black')
english_entry.insert("0.0", "\n\n\n\n                                                                                       ENTER键开始翻译")
english_entry.pack()
english_entry.bind("<Button-1>", button_delete)
english_entry.bind("<Return>", start_translation)


'''中文输出'''
c_label=ttk.Label(fm2, text="中文: ", font=('Calibri', 15))
c_label.pack(side=tkinter.LEFT)
scrollbar_c = tkinter.Scrollbar(fm2)
scrollbar_c.pack(side=tkinter.RIGHT, fill=tkinter.Y)
chinese_entry = tkinter.Text(fm2, font=("Calibri 14"), yscrollcommand=scrollbar_v.set, wrap=tkinter.CHAR)
chinese_entry.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


root.mainloop()