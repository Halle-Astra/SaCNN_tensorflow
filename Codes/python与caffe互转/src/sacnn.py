# -*- coding: utf-8 -*-

import tensorflow as tf
import src.layers as L
import numpy as np

csz = (1/8)#change size 的比例
#out_art = 512
out_art = 512
out1 = 64#16
out2 = 64#32

def backbone(input_tensor):#based on VGG16 
    x = L.vgg_conv(input_tensor,'conv1_1')
    x = L.vgg_conv(x,'conv1_2')
    x = L.vgg_pool(x,'pool1_1')
    x = L.vgg_conv(x,'conv2_1')
    x = L.vgg_conv(x,'conv2_2')
    x = L.vgg_pool(x,'pool2_1')
    x = L.vgg_conv(x,'conv3_1')
    x = L.vgg_conv(x,'conv3_2')
    x = L.vgg_conv(x,'conv3_3')
    x = L.vgg_pool(x,'pool3_1')
    x = L.vgg_conv(x,'conv4_1')#,trainable = True)
    x = L.vgg_conv(x,'conv4_2')#,trainable = True)
    x = L.vgg_conv(x,'conv4_3')#,trainable = True)
    return x

def subscale_1(x):
    net = L.pool(x,name = 'pool4',kh=2,kw=2,dw=2,dh=2)
    net = L.conv(net,name = 'conv5_1',kh = 3,kw = 3,n_out = out_art)
    net = L.conv(net,name = 'conv5_2',kh = 3,kw = 3,n_out = out_art)
    net = L.conv(net,name = 'conv5_3',kh = 3,kw = 3,n_out = out_art)
    return net

def subscale_2(x):
    x = L.pool(x,name = 'pool5',kh=3,kw=3,dw=1,dh=1)
    net = L.conv(x,name = 'conv6_1',kh = 3,kw = 3,n_out = out_art)
    return net

def fuse_layer(x1, x2):
    x_concat = tf.concat([x1, x2],axis=3)
    return x_concat

def build(input_tensor):
    net1 = backbone(input_tensor)
    net2 = subscale_1(net1)
    net3 = subscale_2(net2)
    fuse1 = fuse_layer(net2,net3)
    h3 = tf.shape(net1)[1]
    w3 = tf.shape(net1)[2]
    fuse1 = L.model_deconv(fuse1,'conv_concat1_2x',
                   out_shape = [1,h3,w3,1])#[1,?,?,1024]
    fuse2 = fuse_layer(net1,fuse1)
    fuse2 = L.conv(fuse2,name = 'p_conv1',kh = 3,kw = 3,n_out = out_art)
    fuse2 = L.conv(fuse2,name = 'p_conv2',kh = 3,kw = 3,n_out = 256)
    fuse2 = L.conv(fuse2,name = 'p_conv3',kh = 1,kw = 1,n_out = 1,activation_fn = None)
    return fuse2