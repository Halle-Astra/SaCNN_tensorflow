# -*- coding: utf-8 -*-

import tensorflow as tf 
from matplotlib import pyplot as plt 
import numpy as np 
import os 
import time 
from layers import * 
        
def backbone(x):#based on vgg16
    x = model_conv(x,'conv1_1')
    x = model_conv(x,'conv1_2')
    x = model_pool(x,'pool1_1')
    x = model_conv(x,'conv2_1')
    x = model_conv(x,'conv2_2')
    x = model_pool(x,'pool2_1')
    x = model_conv(x,'conv3_1')
    x = model_conv(x,'conv3_2')
    x = model_conv(x,'conv3_3')
    x = model_pool(x,'pool3_1')
    x = model_conv(x,'conv4_1')
    x = model_conv(x,'conv4_2')
    x = model_conv(x,'conv4_3')
    return x

def subscale_1(x):
    x = model_pool(x,'pool4')
    x = model_conv(x,'conv5_1')
    x = model_conv(x,'conv5_2')
    x = model_conv(x,'conv5_3')
    return x 

def subscale_2(x):
    x = pool(x,name = 'pool5',kh=3,kw=3,dw=1,dh=1)
    x = model_conv(x,name = 'conv6_1')
    return x

def build(input_tensor):
    net1 = backbone(input_tensor)
    net2 = subscale_1(net1)
    net3 = subscale_2(net2)
    fuse1 = fuse_layer(net2,net3)
    h3 = tf.shape(net1)[1]
    w3 = tf.shape(net1)[2]
    fuse1 = model_deconv(fuse1,'Deconv1',
                   out_shape = [1,h3,w3,1])#[1,?,?,1024]
    fuse2 = fuse_layer(net1,fuse1)
    fuse2 = model_conv(fuse2,'p_conv1')
    fuse2 = model_conv(fuse2,'p_conv2')
    fuse2 = model_conv(fuse2,'p_conv3',activation = None)
    return fuse2