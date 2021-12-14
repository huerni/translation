from tkinter import *
import tkinter as tk
import easygui
import requests

master = Tk()
from tkinter import ttk
ttk.Label(master,text='标题:').grid(row=0,column=0)
e1 =  ttk.Entry(master,
            width=44,
            font=('StSong', 14),
            foreground='green')
e1.grid(row=1)
ttk.Label(master,text='内容:').grid(row=2,column=0)

e2 = Text(master,
            width=44,
            height=4,
            font=('StSong', 14),
            foreground='gray')
e2.grid(row=3,column=0)
# 创建Scrollbar组件，设置该组件与text2的纵向滚动关联
scroll = Scrollbar(master, command=e2.yview)
scroll.grid(row = 3,sticky = tk.N + tk.S+tk.E)
#scroll.pack(side=RIGHT, fill=Y)
## 设置text2的纵向滚动影响scroll滚动条
e2.configure(yscrollcommand=scroll.set)

def calc():
    if e1.get()=='':
        easygui.msgbox('请先输入内容 !')
        return
    if e2.get(1.0, END)=='':
        easygui.msgbox('请先输入内容 !')
        return
    user_info = {'title':e1.get(),
                 'content':e2.get(1.0, END)}
    r = requests.post("http://127.0.0.1:5000/cluster/", data=user_info)
    e3.insert(INSERT, r.text)

Button(master,text='提交',command=calc,width=10, height=1).grid(row = 4)
e3 = Text(master,
            width=44,
            height=4,
            font=('StSong', 14),
            foreground='gray')
e3.grid(row = 5)

# 创建Scrollbar组件，设置该组件与text2的纵向滚动关联
scroll = Scrollbar(master, command=e3.yview)
scroll.grid(row = 5,sticky = tk.N + tk.S+tk.E)
e3.configure(yscrollcommand=scroll.set)

mainloop()