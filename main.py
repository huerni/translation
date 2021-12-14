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
root.geometry("%dx%d+%d+%d" %(1000, 600, (w/2)-500, (h/2)-300))
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
	english_value = preprocess_sentence(english.get())
	translator = str.maketrans('', '', string.punctuation)
	english_value = english_value.translate(translator)
	english_value = [str(english_value)]
	english_value = eng_tokenizer.texts_to_sequences(english_value)
	eng_padded = pad_sequences(english_value, maxlen=109, padding='post')
	eng_padded = np.array(eng_padded)
	chinese.set(get_predicted_sentence(eng_padded[0].reshape(1, 109))[:-4])

e_label=ttk.Label(fm1, text="英语: ", font=('Calibri', 15))
e_label.pack(side=tkinter.LEFT)
english = tkinter.StringVar()
english_entry = tkinter.Text(fm1, font=("Calibri 14"))
#english_entry.insert(0, '回车键开始翻译')
english_entry.pack(side=tkinter.TOP,  fill=tkinter.BOTH,  expand = 1)
english_entry.bind("<Return>", start_translation)

c_label=ttk.Label(fm2, text="中文: ", font=('Calibri', 15))
c_label.pack(side=tkinter.LEFT)
chinese = tkinter.StringVar()
chinese_entry = tkinter.Text(fm2, font=("Calibri 14"),  state='disabled')
chinese_entry.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


root.mainloop()