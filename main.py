import matplotlib
matplotlib.use('TkAgg')
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import networkx as nx
import random
from tkinter import ttk

root = tkinter.Tk()  # 创建tkinter的主窗口
root.tk.call("source", "azure.tcl")
root.tk.call("set_theme", "light")
root.wm_title("Translation")
w = root.winfo_screenwidth()
h = root.winfo_screenheight()
root.geometry("%dx%d+%d+%d" %(800, 400, (w/2)-400, (h/2)-200))
root.configure(bg='white')

fm1 = ttk.Frame(root)
fm1.pack(side=tkinter.TOP,  # 上对齐
		fill=tkinter.BOTH,  # 填充方式
		expand=1)

fm12 = ttk.Frame(root)
fm12.pack(side=tkinter.TOP)
fm2 = ttk.Frame(root)
fm2.pack(side=tkinter.TOP,fill=tkinter.BOTH, expand=1)

def start_translation(event, *args):
	print(english.get())
	chinese.set(english.get())

e_label=ttk.Label(fm1,text="英语: ", font=('Calibri', 15))
e_label.pack(side=tkinter.LEFT)
english = tkinter.StringVar()
english_entry = ttk.Entry(fm1, font=("Calibri 12"),textvariable = english,)
english_entry.insert(0, '回车键开始翻译')
english_entry.pack(side=tkinter.TOP,  fill=tkinter.BOTH,  expand=1)
english_entry.bind("<Return>", start_translation)

c_label=ttk.Label(fm2,text="中文: ", font=('Calibri', 15))
c_label.pack(side=tkinter.LEFT)
chinese = tkinter.StringVar()
chinese_entry = ttk.Entry(fm2, font=("Calibri 12"), textvariable = chinese, state='disabled')
chinese_entry.pack(side=tkinter.TOP,fill=tkinter.BOTH, expand=1)


root.mainloop()