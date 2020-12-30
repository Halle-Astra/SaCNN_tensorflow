# -*- coding: utf-8 -*-

'''本脚本用于存放进行caffe与tf进行比较的一些预备函数
从最后的结果来看，减去均值根本没有生效,但是通道交换是有效的
'''
import numpy as np 
from matplotlib import pyplot as plt 

VGG_MEAN = np.load(r'D:\Medium\Codes\Caffe_build\caffe\python\caffe\imagenet\ilsvrc_2012_mean.npy').mean(1).mean(1)
img_test_path = r'D:\Medium\Codes\SACNN\sacnn-adaptived\data\comb_dataset_v3\part_B_final\train_data\images\IMG_1.jpg'
#在进行mean之前，np.load得到的是shape为[3,256,256]的东西来着。
caffe_path = 'caffe_blobs/'

caffe_dict = np.load(caffe_path+'comp_caffe.npy',allow_pickle=True).item()

def image_caffe2tf(transformed_image):
    '''输入直接就是从caffe_dict里得到的四维tensor即可'''
    img_tt = np.zeros((transformed_image.shape[2],transformed_image.shape[3],transformed_image.shape[1]))
    img_tt[:,:,0] = transformed_image[0,2,:,:]
    img_tt[:,:,1] = transformed_image[0,1,:,:]
    img_tt[:,:,2] = transformed_image[0,0,:,:]
    return img_tt

def load_imgtf_eg():
    '''因为是改变caffe来和tf的结果做对比，所以这里作为tf的读入图像，应该是不需要交换通道的'''
    img = plt.imread(r'D:\Medium\Codes\SACNN\sacnn-adaptived\data\comb_dataset_v3\part_B_final\train_data\images\IMG_1.jpg')
    img_res = np.zeros(img.shape)
    img_res[:,:,0] = img[:,:,0]-VGG_MEAN[2]
    img_res[:,:,1] = img[:,:,1]-VGG_MEAN[1]
    img_res[:,:,2] = img[:,:,2]-VGG_MEAN[0]
    return img_res

