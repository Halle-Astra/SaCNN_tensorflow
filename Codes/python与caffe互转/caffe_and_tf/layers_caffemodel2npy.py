# -*- coding: utf-8 -*-

import tensorflow as tf 
import numpy as np 

net_model = np.load('sacnn_tf.npy',allow_pickle = True).item()
layer_names = ['conv1_1', 'conv1_2', 'conv2_1',
               'conv2_2', 'conv3_1', 'conv3_2',
               'conv3_3', 'conv4_1', 'conv4_2',
               'conv4_3', 'conv5_1', 'conv5_2',
               'conv5_3', 'conv6_1', 'conv_concat1_2x', 
               'p_conv1', 'p_conv2', 'p_conv3']

def model_conv(x,name,trainable = False,activation = tf.nn.relu):
    with tf.variable_scope(name):
        if trainable :
            gene_fn = tf.Variable
        else:
            gene_fn = tf.constant
        y = tf.nn.bias_add(
            tf.nn.conv2d(x,gene_fn(net_model[name][0],dtype = tf.float32),
                         (1,1,1,1),padding = 'SAME',name = name),
            gene_fn(net_model[name][1],dtype = tf.float32))
        if activation is not None:
            print(name,'activate')
            print(name,'sum is ',net_model[name][1].sum(),net_model[name][1].max())
            return activation(y)
        else:
            return y
    
def model_pool(x,name):
    with tf.variable_scope(name):
        return tf.nn.max_pool(x,
                          ksize=[1,2, 2 , 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME',
                          name=name)
        
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
    return tf.nn.max_pool(input_tensor,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)

def fuse_layer(x1, x2):
    x_concat = tf.concat([x1, x2],axis=3)
    return x_concat