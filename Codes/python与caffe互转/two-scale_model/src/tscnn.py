
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

def second_scale(x):
    net = L.pool(x,name = 'pool4',kh=2,kw=2,dw=1,dh=1)
    #net_mean = L.mean_pool(x,name = 'pool4',kh=2,kw=2,dw=1,dh=1)
    #net = net-net_mean
    net = L.conv(net,name = 'conv5_1',kh = 3,kw = 3,n_out = out_art)
    net = L.conv(net,name = 'conv5_2',kh = 3,kw = 3,n_out = out_art)
    net = L.conv(net,name = 'conv5_3',kh = 3,kw = 3,n_out = out_art)
    return net#,net_mean

def last_layer(x):
    net = L.conv(x,name = 'p_conv_1',kh = 1,kw = 1,n_out = 2*out_art)#,activation_fn = tf.nn.leaky_relu)#out_art)
    net = L.conv(net,name = 'p_conv_2',kh = 1,kw = 1,n_out = 2*out_art)
    #net = L.conv(net,name = 'p_conv_3',kh = 1,kw = 1,n_out = out_art)
    net = L.conv(net,name = 'p_conv_3',kh = 1,kw = 1,n_out = 1,activation_fn = tf.nn.relu)
    #net = 0.00001*net
    return net

def fuse_layer(x1, x2):
    x_concat = tf.concat([x1, x2],axis=3)
    return x_concat


def build(input_tensor, norm = False):
    tf.summary.image('input', input_tensor, 3)
    if norm:
        input_tensor = tf.cast(input_tensor, tf.float32) * (1. / 255) - 0.5
    net1 = backbone(input_tensor)#去掉norm后依然没有梯度
    #with tf.variable_scope('batch_norm_1'):
    #    net1 = tf.contrib.layers.batch_norm(net1)#到这就已经没有梯度了
    net2 = second_scale(net1)#这里梯度也是0
    with tf.variable_scope('batch_norm_2'):
        net2 = tf.contrib.layers.batch_norm(net2)
    fuse1 = fuse_layer(net1,net2)#梯度还是0
    d_map = last_layer(fuse1)#0.2左右的梯度
    return d_map,d_map


# Testing the data flow of the network with some random inputs.
if __name__ == "__main__":
    x = tf.placeholder(tf.float32, [1, 200, 300, 1])
    net = build(x)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    d_map = sess.run(net,feed_dict={x:255*np.ones(shape=(1,200,300,1), dtype=np.float32)})
    prediction = np.asarray(d_map)
    prediction = np.squeeze(prediction, axis=0)
    prediction = np.squeeze(prediction, axis=2)
