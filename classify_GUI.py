import tkinter.ttk
from tkinter import *
from tkinter import filedialog
import torch
from torchvision import transforms
import pandas as pd
from torchvision import models
import numpy as np
from PIL import Image, ImageTk
import tkinter.font as tkFont
from tkinter import ttk

"""Written by Cai"""
with torch.no_grad():
    df = pd.read_csv('label.csv')

    net_list = {'resnet18': models.resnet18(pretrained=True), 'resnet50': models.resnet50(pretrained=True),
                'resnet101': models.resnet101(pretrained=True),
                'mobilenetv2': models.mobilenet_v2(pretrained=True),
                'mobilenetv3_l': models.mobilenet_v3_large(pretrained=True),
                'mobilenetv3_s': models.mobilenet_v3_small(pretrained=True),
                'vgg11': models.vgg11_bn(pretrained=True), 'vgg16': models.vgg16_bn(pretrained=True),
                'googlenet': models.googlenet(pretrained=True),
                'efficient_v1': models.efficientnet_b2(pretrained=True),
                'efficient_v2_l': models.efficientnet_v2_l(pretrained=True),
                'efficient_v2_m': models.efficientnet_v2_m(pretrained=True),
                'efficient_v2_s': models.efficientnet_v2_s(pretrained=True)}

    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])

    img_file = None
    path = None
    def open_pred():
        global img_file
        global path
        ori_path = path
        path = filedialog.askopenfilename()
        if path == '':
            path = ori_path
            return

        img = Image.open(path)
        img = trans(img)
        img_file = Image.open(path)
        w, h = img_file.size
        if w / h > show_img.winfo_width() / (show_img.winfo_height() * 11 / 12):
            img_file = img_file.resize((show_img.winfo_width(), int(show_img.winfo_width() / w * h)))
        else:
            img_file = img_file.resize(
                (int(11 / 12 * show_img.winfo_height() / h * w), int(11 / 12 * show_img.winfo_height())))
        img_file = ImageTk.PhotoImage(img_file)
        show_img.create_image(int(show_img.winfo_width() * 0.5), 0, anchor='n', image=img_file)
        net = net_list[bbox.get()]
        net.eval()
        prediction = torch.softmax(net(img.reshape(1, 3, img.shape[1], -1)), dim=1).detach().numpy()
        label = np.argmax(prediction, axis=1)

        var.set(df.iloc[label, 1].values[0] + f':{np.max(prediction):.1%}')


    def pred():
        global img_file
        global path
        img = Image.open(path)
        img = trans(img)
        img_file = Image.open(path)
        w, h = img_file.size
        if w / h > show_img.winfo_width() / (show_img.winfo_height() * 11 / 12):
            img_file = img_file.resize((show_img.winfo_width(), int(show_img.winfo_width() / w * h)))
        else:
            img_file = img_file.resize(
                (int(11 / 12 * show_img.winfo_height() / h * w), int(11 / 12 * show_img.winfo_height())))
        img_file = ImageTk.PhotoImage(img_file)
        show_img.create_image(int(show_img.winfo_width() * 0.5), 0, anchor='n', image=img_file)
        net = net_list[bbox.get()]
        net.eval()
        prediction = torch.softmax(net(img.reshape(1, 3, img.shape[1], -1)), dim=1).detach().numpy()
        label = np.argmax(prediction, axis=1)

        var.set(df.iloc[label, 1].values[0] + f':{np.max(prediction):.1%}')


    root = Tk()
    root.title('Pic Predict GUI')
    root.geometry("800x600")
    var = StringVar()

    open_Style = tkFont.Font(family="Lucida Grande", size=12)
    pre_Style = tkFont.Font(family="Bahnschrift SemiBold", size=16)
    # 打开图片的按钮
    open_img = Button(root, text='选择图片', command=open_pred, font=open_Style)
    open_img.place(relx=0.1, rely=0.8, anchor='nw', relwidth=1 / 8, relheight=1 / 12)

    pred_button = Button(root, text='预测', command=pred, font=open_Style)
    pred_button.place(relx=0.48, rely=0.8, anchor='ne', relwidth=1 / 8, relheight=1 / 12)

    show_img = Canvas(root, borderwidth=2, relief='sunken')
    show_img.place(relx=0.06, rely=1 / 15, anchor='nw', relwidth=0.45, relheight=0.7)

    # 设置类别标签
    cls_label = Label(root, text='模型选择: ', font=pre_Style)
    cls_label.place(relx=0.53, rely=0.06, anchor='nw', relwidth=0.15, relheight=0.08)

    # 设置下拉框
    bbox = ttk.Combobox(root,
                        values=['resnet18', 'resnet50', 'resnet101', 'mobilenetv2', 'mobilenetv3_l', 'mobilenetv3_s',
                                'vgg11', 'vgg16', 'googlenet', 'efficient_v1', 'efficient_v2_l', 'efficient_v2_m',
                                'efficient_v2_s'], font=pre_Style)
    bbox.set('resnet18')
    bbox.place(relx=0.69, rely=0.06, anchor='nw', relwidth=0.3, relheight=0.08)

    pre_label = Label(root, text='预测结果: ', font=pre_Style)
    pre_label.place(relx=0.53, rely=0.2, anchor='nw', relwidth=0.15, relheight=0.08)

    # 显示预测概率
    predictive_probability = Label(root, textvariable=var, font=pre_Style, borderwidth=5, relief='groove')
    predictive_probability.place(relx=0.69, rely=0.2, anchor='nw', relwidth=0.3, relheight=0.08)

    root.mainloop()
