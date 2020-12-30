# -*- coding: utf-8 -*-

import os 
from predict_caffe import predict 
from matplotlib import pyplot as plt 
import time 
from PIL import Image 
import numpy as np 

path = r'D:\Medium\Codes\SACNN_application\images/'
fs = os.listdir(path)
fs = [path + i for i in fs ]
imgs = [plt.imread(i) for i in fs]
#imgs = [np.array(Image.fromarray(i).resize((768,1024))) for i in imgs ]
for i in range(len(fs)):
    pred = predict(fs[i])
    fig = plt.figure(figsize = (16,8))
    ax = fig.add_subplot(121)
    ax.imshow(imgs[i])
    ax = fig.add_subplot(122)
    ax.imshow(pred)
    plt.title(pred.sum())
    plt.show()