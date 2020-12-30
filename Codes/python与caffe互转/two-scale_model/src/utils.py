

import glob
import os
import random
import numpy as np
import tensorflow as tf
import sys
import cv2
#import src.layers as L
from PIL import Image 

VGG_MEAN = [103.939, 116.779, 123.68]

def img_one2four(img,img_type = 'data'):
    '''对图像进行切割，一分为四'''
    if img_type == 'label':
        img = np.expand_dims(img,axis = [0,3])
    img_1 = img[:,:384,:512,:]#这里会变成[1,384,512,3]，仅对于上科大数据集
    img_2 = img[:,:384,512:,:]
    img_3 = img[:,384:,:512,:]
    img_4 = img[:,384:,512:,:]
    if img_type == 'label':
        return [i[0,:,:,0] for i in [img_1,img_2,img_3,img_4]]
    return [img_1,img_2,img_3,img_4]

def imgpreprocess(img):
    #img = Image.fromarray(img.astype(np.uint8)).resize((512,384))
    #img = np.array(img)
    img_res = np.zeros(img.shape)
    img_res[:,:,0] = img[:,:,2]-VGG_MEAN[0]
    img_res[:,:,1] = img[:,:,1]-VGG_MEAN[1]
    img_res[:,:,2] = img[:,:,0]-VGG_MEAN[2]
    return img_res#.astype(np.float32)

def get_density_map_gaussian(points, d_map_h, d_map_w):

	im_density = np.zeros(shape=(d_map_h,d_map_w), dtype=np.float32)

	if np.shape(points)[0] == 0:  #如果输入的数据为空
		sys.exit()

	for i in range(np.shape(points)[0]):  #遍历每一条记录

		f_sz = 15
		sigma = 4

		gaussian_kernel = get_gaussian_kernel(f_sz, f_sz, sigma)		 #得到fsize，长宽为15的高斯核
					
		x = min(d_map_w, max(1, np.abs(np.int32(np.floor(points[i, 0])))))
		y = min(d_map_h, max(1, np.abs(np.int32(np.floor(points[i, 1])))))

		if(x > d_map_w or y > d_map_h):
			continue

		x1 = x - np.int32(np.floor(f_sz / 2))
		y1 = y - np.int32(np.floor(f_sz / 2))
		x2 = x + np.int32(np.floor(f_sz / 2))
		y2 = y + np.int32(np.floor(f_sz / 2))

		dfx1 = 0
		dfy1 = 0
		dfx2 = 0
		dfy2 = 0

		change_H = False

		if(x1 < 1):
			dfx1 = np.abs(x1)+1
			x1 = 1
			change_H = True

		if(y1 < 1):
			dfy1 = np.abs(y1)+1
			y1 = 1
			change_H = True

		if(x2 > d_map_w):
			dfx2 = x2 - d_map_w
			x2 = d_map_w
			change_H = True

		if(y2 > d_map_h):
			dfy2 = y2 - d_map_h
			y2 = d_map_h
			change_H = True

		x1h = 1+dfx1
		y1h = 1+dfy1
		x2h = f_sz - dfx2
		y2h = f_sz - dfy2

		if (change_H == True):
			f_sz_y = np.double(y2h - y1h + 1)
			f_sz_x = np.double(x2h - x1h + 1)

			gaussian_kernel = get_gaussian_kernel(f_sz_x, f_sz_y, sigma)

		im_density[y1-1:y2,x1-1:x2] = im_density[y1-1:y2,x1-1:x2] +  gaussian_kernel
	return im_density

def get_gaussian_kernel(fs_x, fs_y, sigma):
	gaussian_kernel_x = cv2.getGaussianKernel(ksize=np.int(fs_x), sigma=sigma)
	gaussian_kernel_y = cv2.getGaussianKernel(ksize=np.int(fs_y), sigma=sigma)
	gaussian_kernel = gaussian_kernel_y * gaussian_kernel_x.T
	return gaussian_kernel

def compute_abs_err(pred, gt):
	"""
	Computes mean absolute error between the predicted density map and ground truth
	:param pred: predicted density map
	:param gt: ground truth density map
	:return: abs |pred - gt|
	"""
	return np.abs(np.sum(pred[:]) - np.sum(gt[:]))

def create_session(log_dir, session_id):
	folder_path = os.path.join(log_dir, 'session-'+str(session_id))
	if os.path.exists(folder_path):
		print ('Session already taken. It will create a different session id.')#所以最好每次删了logs文件夹
		
		#sys.exit()
	else:
		os.makedirs(folder_path)
	return folder_path

def get_file_id(filepath):
	return os.path.splitext(os.path.basename(filepath))[0]

def get_data_list(data_root, mode='train'):

	if mode == 'train':
		imagepath = os.path.join(data_root, 'train_data', 'images')
		gtpath = os.path.join(data_root, 'train_data', 'ground_truth')

	elif mode == 'valid':
		imagepath = os.path.join(data_root, 'valid_data', 'images')
		gtpath = os.path.join(data_root, 'valid_data', 'ground_truth')

	else:
		imagepath = os.path.join(data_root, 'test_data', 'images')
		gtpath = os.path.join(data_root, 'test_data', 'ground_truth')

	image_list = [file for file in glob.glob(os.path.join(imagepath,'*.jpg'))]
	gt_list = []

	for filepath in image_list:
		file_id = get_file_id(filepath)
		gt_file_path = os.path.join(gtpath, 'GT_'+ file_id + '.mat')
		gt_list.append(gt_file_path)

	xy = list(zip(image_list, gt_list))		#列表中为元组
	random.shuffle(xy)		   #会更新数据
	s_image_list, s_gt_list = zip(*xy)	#zip(*)可理解为解压

	return s_image_list, s_gt_list

def reshape_tensor(tensor,channel):
	r_tensor = np.reshape(tensor, newshape=(1, tensor.shape[0], tensor.shape[1], channel))	#
	return r_tensor

def save_weights(graph, fpath):
	sess = tf.get_default_session()
	variables = graph.get_collection("variables")
	variable_names = [v.name for v in variables]
	kwargs = dict(zip(variable_names, sess.run(variables)))
	np.savez(fpath, **kwargs)

def load_weights(graph, fpath):
	sess = tf.get_default_session()
	variables = graph.get_collection("variables")
	data = np.load(fpath)
	for v in variables:
		if v.name not in data:
			print("could not load data for variable='%s'" % v.name)
			continue
		print("assigning %s" % v.name)
		sess.run(v.assign(data[v.name]))

def labelmap(img,loc):
	img[loc.astype('int')] = 255
	return img

def resize_gt(train_d_map_r):
    '''resize ground truth'''
    img = Image.fromarray(train_d_map_r)
    #img = img.resize(compute_downsampling(512,384))
    img = img.resize(compute_downsampling(train_d_map_r.shape[1],train_d_map_r.shape[0]))
    img = np.array(img)
    real_sum = train_d_map_r.sum()
    if real_sum!=0:
        factor = train_d_map_r.sum()/img.sum()
    else:
        factor = 0
    return img*factor

def compute_downsampling(h,w):
    for i in range(3):
        if h%2 == 0:
            h = h/2
        else:
            h = np.ceil(h/2)
        if w%2 == 0:
            w = w/2
        else:
            w = np.ceil(w/2)
    return int(h),int(w)

'''def downsampling_d_map(train_d_map_r,sess):
	train_d_map_r = L.pool(train_d_map_r,name = 'ds1',kh = 2,kw = 2,dh = 2,dw = 2)
	train_d_map_r = L.pool(train_d_map_r,name = 'ds2',kh = 2,kw = 2,dh = 2,dw = 2)
	train_d_map_r = L.pool(train_d_map_r,name = 'ds3',kh = 2,kw = 2,dh = 2,dw = 2)
	train_d_map_r = sess.run(train_d_map_r)
	return train_d_map_r'''