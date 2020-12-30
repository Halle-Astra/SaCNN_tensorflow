

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
    
def deconv(input_tensor,name ,kw,kh,n_out,out_shape ,dw=2,dh=2,activation_fn=tf.nn.leaky_relu):
    "这是反卷积"
    n_in = input_tensor.get_shape()[-1].value

    #因为直接作为numpy数组来的话，似乎全都拿不到，这样的话，我至少自己指定最后一维，第一维他会自己拿到，但，，emm我怀疑会出问题
    
    with tf.variable_scope(name):
        weights = tf.Variable(tf.truncated_normal(shape = (kh,kw,n_out,n_in),stddev = 0.01),dtype = tf.float32,name = 'weights')
        #biases = tf.Variable(tf.constant(0.0,shape = [n_out]),dtype = tf.float32,name = 'biases')
        deconv_t = tf.nn.conv2d_transpose(input_tensor,weights,out_shape ,strides = (1,dh,dw,1),padding = 'SAME' )
    #conv2d_transpose(value, filter, output_shape, strides, padding="SAME", 
    #              data_format="NHWC", name=None)
        return deconv_t
    
def pool(input_tensor, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_tensor,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)
    
def mean_pool(input_tensor, name, kh, kw, dh, dw):
    return tf.nn.avg_pool(input_tensor,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)

def loss(est, gt):
    return  0*tf.reduce_mean(tf.pow((tf.reduce_max(est,axis = [1,2,3])-tf.reduce_max(gt,axis = [1,2,3])),2))+\
                           1*tf.reduce_mean(tf.reduce_sum(tf.pow((est-gt),2),axis = [1,2,3]))+\
                0.1*tf.reduce_mean(tf.pow(((tf.reduce_sum(est,axis = [1,2,3])-\
                                         tf.reduce_sum(gt,axis = [1,2,3]))/(tf.reduce_sum(gt,axis = [1,2,3])+1)),2))
    #return tf.losses.mean_squared_error(est, gt)       #所以这一步是绝对没法做loss的      est为 (1,192,256,1)  gt为(1,33,2,1)
    
# Module to test the loss layer
if __name__ == "__main__":
    x = tf.placeholder(tf.float32, [1, 20, 20, 1])
    y = tf.placeholder(tf.float32, [1, 20, 20, 1])
    mse = loss(x, y)
    sess = tf.Session()
    dict = {
        x: 5*np.ones(shape=(1,20,20,1)),
        y: 4*np.ones(shape=(1,20,20,1))
    }
    print (sess.run(mse, feed_dict=dict) )
