from tkinter import *
from tkinter import filedialog
import torch
from torchvision import transforms
import pandas as pd
from torchvision import models
import numpy as np
from PIL import Image, ImageTk
import tkinter.font as tkFont


df = pd.read_csv('label.csv')
# 14行加载自己的训练好的模型，15行使用自己的预处理方式
net = models.resnet50(pretrained=True)
trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])


img_file = None
def predict():
    global img_file
    path = filedialog.askopenfilename()
    if path == '':
        return

    img = Image.open(path)
    img = trans(img)
    img_file = Image.open(path)
    w, h = img_file.size
    if w/h > root.winfo_width()/(root.winfo_height()*11/12):
        img_file = img_file.resize((root.winfo_width(), int(root.winfo_width() / w * h)))
    else:
        img_file = img_file.resize((int(11/12*root.winfo_height() / h * w), int(11/12*root.winfo_height())))
    img_file = ImageTk.PhotoImage(img_file)
    show_img.create_image(int(root.winfo_width()*0.5), 0, anchor='n', image=img_file)
    net.eval()
    prediction = torch.softmax(net(img.reshape(1, 3, img.shape[1], -1)), dim=1).detach().numpy()
    label = np.argmax(prediction, axis=1)

    var.set(df.iloc[label, 1].values[0] + f':{np.max(prediction):.2%}')


root = Tk()
root.title('pic predict')
root.geometry("800x600")
var = StringVar()

open_Style = tkFont.Font(family="Lucida Grande", size=12)
pre_Style = tkFont.Font(family="Bahnschrift SemiBold", size=16)
# 打开图片的按钮
open_img = Button(root, text='打开图片', command=predict,font=open_Style)
open_img.place(relx=0, rely=0, anchor='nw', relwidth=1/8, relheight=1/12)
# 显示预测概率
predictive_probability = Label(root, textvariable=var, bg='pink',font=pre_Style)
predictive_probability.place(relx=100/800, y=0, anchor='nw', relwidth=7/8, relheight=1/12)

show_img = Canvas(root)
show_img.place(relx=0.5, rely=1/15, anchor='n', relwidth=1, relheight=11/12)


root.mainloop()
