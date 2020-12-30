# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import numpy as np
import  pylab as plt
import keras.backend.tensorflow_backend as KTF
from SaCNN_tf import *
import time
from matplotlib import pyplot as plt 
from PIL import Image

VGG_MEAN = [103.939, 116.779, 123.68]
#VGG_MEAN = [ 104.00698793,  116.66876762,  122.67891434]
img_side_max = 1200#177,79
def imgpreprocess(img):
    if img.shape[0] > img_side_max or img.shape[1] > img_side_max:
        img_t = Image.fromarray(img)
        if img.shape[0] == max(img.shape):
            img_t = img_t.resize((int(img.shape[1]*img_side_max/img.shape[0]),img_side_max))
            img = np.array(img_t)
        else:
            img_t = img_t.resize((img_side_max,int(img.shape[0]*img_side_max/img.shape[1])))
            img = np.array(img_t)
    img_res = np.zeros(img.shape)
    img_res[:,:,0] = img[:,:,2]-VGG_MEAN[0]
    img_res[:,:,1] = img[:,:,1]-VGG_MEAN[1]
    img_res[:,:,2] = img[:,:,0]-VGG_MEAN[2]
    return np.expand_dims(img_res,axis = 0)

def img_preprocess(img):
    if img.shape[0] > img_side_max or img.shape[1] > img_side_max:
        img_t = Image.fromarray(img)
        if img.shape[0] == max(img.shape):
            img_t = img_t.resize((int(img.shape[1]*img_side_max/img.shape[0]),img_side_max))
            img = np.array(img_t)
        else:
            img_t = img_t.resize((img_side_max,int(img.shape[0]*img_side_max/img.shape[1])))
            img = np.array(img_t)
    img_res = np.zeros(img.shape)
    img_res[:,:,0] = img[:,:,2]
    img_res[:,:,1] = img[:,:,1]
    img_res[:,:,2] = img[:,:,0]
    return np.expand_dims(img_res,axis = 0)

if __name__ == "__main__":    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    input_path = 'images/'
    dmap_path = 'dmaps/'
    for i in [input_path,dmap_path]:
        if not os.path.exists(i):
            os.mkdir(i)
    G = tf.Graph()
    with G.as_default():
        image_place_holder = tf.placeholder(tf.float32, shape=[None, None, None, 3])     
        d_map_est = build(image_place_holder)      
       
        summary = tf.summary.merge_all()

        with tf.Session(graph=G,config=config) as sess:
            KTF.set_session(sess)
            writer = tf.summary.FileWriter(os.path.join('logging'))
            writer.add_graph(sess.graph)
            t_old = time.time()
            time.sleep(1)
            while (time.time()-t_old)>1:
                t_old = time.time()
                imgs = os.listdir(input_path)
                res_imgs = os.listdir(dmap_path)
                imgs = [input_path+i for i in imgs if os.path.split(i)[-1] not in res_imgs]
                #imgs_net_ls = [imgpreprocess(plt.imread(i)) for i in imgs]
                imgs_net_ls = [img_preprocess(plt.imread(i)) for i in imgs]
                for i in range(len(imgs_net_ls)):
                    imgs_net = imgs_net_ls[i]
                    d_map = sess.run(d_map_est,feed_dict = {image_place_holder:imgs_net})
                    res_t = d_map[0,:,:,0]
                    plt.imshow(res_t)
                    plt.title(res_t.sum())
                    plt.axis('off')
                    print(os.path.split(imgs[i])[-1])
                    plt.savefig(os.path.join(dmap_path,os.path.split(imgs[i])[-1]))
                    np.save(os.path.join(dmap_path,os.path.split(imgs[i])[-1]),res_t)
                    plt.clf()
                    fig = plt.figure(figsize = (16,8))
                    ax = fig.add_subplot(121)
                    ax.imshow(plt.imread(imgs[i]))
                    ax = fig.add_subplot(122)
                    ax.imshow(res_t)
                    plt.title(res_t.sum())
                    plt.show()