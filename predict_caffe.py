import numpy as np
import matplotlib.pyplot as plt
import caffe
import os
import sys
from PIL import Image 

plt.rcParams['image.interpolation'] = 'nearest'  
#plt.rcParams['image.cmap'] = 'gray' 

caffe_root = 'D:\\Medium\\Codes\\Caffe_build\\caffe/'  
sys.path.insert(0, caffe_root + 'python')

if os.path.isfile('ShanghaiTech_part_B.caffemodel'):
    print('CaffeNet found.')
else:
    print('CaffeNet Not found.')
    
caffe.set_mode_cpu()
model_def = 'deploy.prototxt'
model_weights = 'ShanghaiTech_part_B.caffemodel'
net = caffe.Net(model_def,model_weights,caffe.TEST)
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  
print('mean-subtracted values:', zip('BGR', mu))

def data2image(data):
    img = np.zeros((data.shape[2],data.shape[3],data.shape[1]))
    img[:,:,2] = data[0,0,:,:]
    img[:,:,1] = data[0,1,:,:]
    img[:,:,0] = data[0,2,:,:]
    return img 

img_side_max = 1500
def imgadapt(img):
    #try:
    if img.shape[0] > img_side_max or img.shape[1] > img_side_max:
        img_t = Image.fromarray(img)
        if img.shape[0] == max(img.shape):
            img_t = img_t.resize((int(img.shape[1]*img_side_max/img.shape[0]),img_side_max))
            img = np.array(img_t)
        else:
            img_t = img_t.resize((img_side_max,int(img.shape[0]*img_side_max/img.shape[1])))
            img = np.array(img_t)
    #except:
    #    print('Data type is ',img.dtype)
    return img.astype(np.float32)

def predict(imgfile=r'D:\Medium\Codes\SACNN\SaCNN-CrowdCounting-Tencent_Youtu-master\SaCNN-master\data\comb_dataset_v3\part_B_final\train_data\images/IMG_109.jpg'):
    image = caffe.io.load_image(imgfile)
    image = imgadapt(image.astype(np.uint8))
    print(image.shape)
    transformer = caffe.io.Transformer({'data': (1,3,image.shape[0],image.shape[1])})
    transformer.set_transpose('data', (2,0,1))  
    transformer.set_mean('data', mu)            
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))
    net.blobs['data'].reshape(1,3,image.shape[0],image.shape[1])
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    print(imgfile,net.blobs['data'].data.shape)
    out = net.forward()
    print(out['estdmap'].shape)
    output = out['estdmap'].data
    return np.array(output)[0,0,:,:]
    
if __name__ == '__main__':
    d_map = predict()
    plt.imshow(d_map)
    plt.show()