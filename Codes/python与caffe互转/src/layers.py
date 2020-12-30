# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

vgg_param = np.load('vgg16.npy',encoding = 'latin1',allow_pickle = True).item()

def vgg_conv(x,name,trainable = False):
    with tf.variable_scope(name):
        if trainable :
            gene_fn = tf.Variable
        else:
            gene_fn = tf.constant
        return tf.nn.relu(tf.nn.bias_add(
            tf.nn.conv2d(x,gene_fn(vgg_param[name][0],dtype = tf.float32),
                         (1,1,1,1),padding = 'SAME',name = name),
            gene_fn(vgg_param[name][1],dtype = tf.float32)))
            
    
def vgg_pool(x,name):
    with tf.variable_scope(name):
        return tf.nn.max_pool(x,
                          ksize=[1,2, 2 , 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME',
                          name=name)
    
def conv(input_tensor, name, kw, kh, n_out, dw=1, dh=1, activation_fn=tf.nn.relu,trainable = True):
    """
    Convolution layer
    :param input_tensor: Input Tensor (feature map / image)
    :param name: name of the convolutional layer
    :param kw: width of the kernel
    :param kh: height of the kernel
    :param n_out: number of output feature maps
    :param dw: stride across width
    :param dh: stride across height
    :param activation_fn: nonlinear activation function
    :return: output feature map after activation
    """
    n_in = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name):
        if trainable :
            gene_fn = tf.Variable
        else:
            gene_fn = tf.constant
        weights = gene_fn(tf.truncated_normal(shape=(kh, kw, n_in, n_out), mean = 0.0,stddev=0.1), dtype=tf.float32, name='weights')
        biases = gene_fn(tf.constant(0.0, shape=[n_out]), dtype=tf.float32, name='biases')
        conv = tf.nn.conv2d(input_tensor, weights, (1, dh, dw, 1), padding='SAME')
        if activation_fn :
            activation = activation_fn(tf.nn.bias_add(conv, biases))
        else:
            activation = tf.nn.bias_add(conv,biases)
        tf.summary.histogram("weights", weights)
        return activation
    
def model_deconv(input_tensor,name,out_shape ,dw=2,dh=2):
    with tf.variable_scope(name):
        deconv_ls = []
        weights = tf.constant(np.ones((2,2,1,2)),dtype = tf.float32)
        for i in range(input_tensor.shape[3]):
            tensor_t = input_tensor[:,:,:,i*2:(i+1)*2]
            deconv_t = tf.nn.conv2d_transpose(tensor_t,weights,out_shape ,strides = (1,dh,dw,1),padding = 'SAME' )
            deconv_ls.append(deconv_t)
            if (i+1)*2==input_tensor.shape[3]:
                break
        return tf.concat(deconv_ls,axis = 3)
    
def pool(input_tensor, name, kh, kw, dh, dw):
    """
    Max Pooling layer
    :param input_tensor: input tensor (feature map) to the pooling layer
    :param name: name of the layer
    :param kh: height scale down size. (Generally 2)
    :param kw: width scale down size. (Generally 2)
    :param dh: stride across height
    :param dw: stride across width
    :return: output tensor (feature map) with reduced feature size (Scaled down by 2).
    """
    return tf.nn.max_pool(input_tensor,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)
    
def mean_pool(input_tensor, name, kh, kw, dh, dw):
    """
    Max Pooling layer
    :param input_tensor: input tensor (feature map) to the pooling layer
    :param name: name of the layer
    :param kh: height scale down size. (Generally 2)
    :param kw: width scale down size. (Generally 2)
    :param dh: stride across height
    :param dw: stride across width
    :return: output tensor (feature map) with reduced feature size (Scaled down by 2).
    """
    return tf.nn.avg_pool(input_tensor,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)

def loss(est, gt):
    """
    Computes mean square error between the network estimated density map and the ground truth density map.
    :param est: Estimated density map
    :param gt: Ground truth density map
    :return: scalar loss after doing pixel wise mean square error.
    """
    return  0*tf.reduce_mean(tf.pow((tf.reduce_max(est,axis = [1,2,3])-tf.reduce_max(gt,axis = [1,2,3])),2))+\
                           1*tf.reduce_mean(tf.reduce_sum(tf.pow((est-gt),2),axis = [1,2,3]))+\
                0.1*tf.reduce_mean(tf.pow(((tf.reduce_sum(est,axis = [1,2,3])-\
                                         tf.reduce_sum(gt,axis = [1,2,3]))/(tf.reduce_sum(gt,axis = [1,2,3])+1)),2))