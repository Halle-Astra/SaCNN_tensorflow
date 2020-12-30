import tensorflow as tf
import src.tscnn as tscnn
import src.layers as L
import os
import src.utils as utils
import numpy as np
import matplotlib.image as mpimg
import scipy.io as sio
import time
import argparse
#import sys
#from PIL import Image
import  pylab as plt
import keras.backend.tensorflow_backend as KTF
#tf.device('/gpu:0')
# Global Constants. Define the number of images for training, validation and testing.
NUM_TRAIN_IMGS = 6000
NUM_VAL_IMGS = 590
NUM_TEST_IMGS = 587
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'tscnn_model'
batch_size = 4
def main(args):
    
    args.retrain = True
    
    temp_use_path = os.path.join(args.log_dir, 'session-'+str(args.session_id))
    if os.path.exists(temp_use_path):
        session_id_list = [tempath.split('-')[-1] for tempath in os.listdir(args.log_dir)]
        session_idl = [eval(comid) for comid in session_id_list]
        session_id_t = max(session_idl)
        if args.retrain:
            args.session_id = session_id_t
            #args.session_id = 133
        else:
            pass#args.session_id = session_id_t+1
    
    sess_path = utils.create_session(args.log_dir, args.session_id)  # Create a session path based on the session id.
    
    if args.retrain:
        model_list = os.listdir(sess_path)
        model_list = [i for i in model_list if '.npz' in i]
        model_list = [eval(i.split('.')[1]) for i in model_list]
        model_final = max(model_list)
        args.base_model_path = sess_path+f'/weights.{model_final}.npz'
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    G = tf.Graph()
    with G.as_default():
        # Create image and density map placeholder
        image_place_holder = tf.placeholder(tf.float32, shape=[None, None, None, 3])         #原本是1，沙雕东西
        d_map_place_holder = tf.placeholder(tf.float32, shape=[None, None, None, 1])

        # Build all nodes of the network
        d_map_est,net_mid = tscnn.build(image_place_holder)       #为什么这里要将channel设置为1，不是rgb么，难道不是3通道吗

        # Define the loss function.
        with tf.variable_scope('loss'):
            euc_loss = L.loss(d_map_est, d_map_place_holder)

        # Define the optimization algorithm

        #optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)
        optimizer= tf.train.AdamOptimizer(0.01)
        #optimizer = tf.train.RMSPropOptimizer(100)
        # Training node.
        #train_op = optimizer.minimize(euc_loss)
        
        #train_grad = tf.gradients(euc_loss,d_map_est)
        train_grad = tf.gradients(euc_loss,net_mid)
        train_grad = tf.reduce_mean(train_grad)
        # Initialize all the variables.
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # For summary
        summary = tf.summary.merge_all()

        with tf.Session(graph=G,config=config) as sess:
            KTF.set_session(sess)
            writer = tf.summary.FileWriter(os.path.join(sess_path,'training_logging'))
            writer.add_graph(sess.graph)
            sess.run(init)                              #完成初始化

            if args.retrain:
                utils.load_weights(G, args.base_model_path)

            # Start the epochs
            for eph in range(args.num_epochs):#训练的轮数
                grads = []

                # Get the list of train images.
                train_images_list, train_gts_list = utils.get_data_list(args.data_root, mode='train')
                #total_train_loss = 0

                # Loop through all the training images
                train_y,train_x = [],[]
                for img_idx in range(len(train_images_list)):#每张图片 捞出来训练
                    
                    # Load the image and ground truth
                    train_image = np.asarray(mpimg.imread(train_images_list[img_idx]), dtype=np.float32)        #卧槽，读图像时直接读成float32啊啊啊啊
                    train_image = utils.imgpreprocess(train_image)
                    train_d_map = np.asarray(sio.loadmat(train_gts_list[img_idx])['image_info'][0][0][0][0][0], dtype=np.float32)      #为了得到个数据，我是没见过比这更有病的

                    # Reshape the tensor before feeding it to the network  #等价于np.expand_dims
                    train_image_r = utils.reshape_tensor(train_image,3)    #将代码修改为根据第二个参数进行设定第4维度

                    train_d_map_r = utils.get_density_map_gaussian(train_d_map,train_image.shape[0],train_image.shape[1]) #通过高斯卷积（模糊）得到密度图
                    #print('-----------------\n',train_d_map_r.shape,'-----------------------')
                    train_d_map_r = utils.img_one2four(train_d_map_r,img_type='label')
                    train_d_map_r = [utils.resize_gt(i) for i in train_d_map_r]#新增放缩算法
                    train_d_map_r = [utils.reshape_tensor(i,1) for i in train_d_map_r]
                    
                    if img_idx%batch_size!=0 or img_idx == 0:
                        #图像切割，one2four，只有训练时需要这样，其他时候没必要
                        #这里切割的实现将交由utils进行
                        #train_image_1 = train_image_r[:,:384,:512,:]#.reshape((1,384,512,3))#这里得到的shape其实是[1,384,512,3]
                        #train_image_2 = train_image_r[:,:384,512:,:]#.reshape((1,384,512,3))
                        
                        train_x+=utils.img_one2four(train_image_r)
                        train_y+=train_d_map_r
                        continue
                    train_x+=utils.img_one2four(train_image_r)
                    train_y+=train_d_map_r
                    train_image_r = np.concatenate(train_x,axis = 0)
                    train_d_map_r = np.concatenate(train_y,axis = 0)
                    train_x,train_y = [],[]
                    #train_d_map_r = utils.downsampling_d_map(train_d_map_r,sess)
                    
                    # Prepare feed_dict
                    feed_dict_data = {
                        image_place_holder: train_image_r,
                        d_map_place_holder: train_d_map_r,
                    }
                                                                     
                    # Compute the loss for one image.
                    #sess.run(train_op,feed_dict = feed_dict_data)
                    
                    #输出loss使用的两个输出，查看形状
                    if img_idx %1== 0:
                        grad_t,net_output = sess.run([train_grad,net_mid],feed_dict = feed_dict_data)
                        grads.append(grad_t)
                        print(net_output.min(),net_output.max(),net_output.mean())
                        plt.figure()
                        if len(grads)<20:
                            plt.plot(grads)
                        else:
                            plt.plot(grads[15:])
                        plt.title('gradients')
                        plt.figure()
                        plt.imshow(net_output[0,:,:,0])
                        plt.title('net_mid')
                        plt.colorbar()
                        plt.figure()
                        plt.imshow(train_d_map_r[0,:,:,0])
                        plt.title('train_d_map')
                        plt.colorbar()
                        plt.show()
                
         

if __name__ == "__main__":
    parser = argparse.ArgumentParser() #所以个文件使你只能使用cmd进去输入参数并传参给脚本，并不，有默认参数

    parser.add_argument('--retrain', default=False, type=bool)
    parser.add_argument('--base_model_path', default=None, type=str)
    parser.add_argument('--log_dir', default = './logs', type=str)
    parser.add_argument('--num_epochs', default = 200, type=int)
    parser.add_argument('--learning_rate', default = 0.0001, type=float)
    parser.add_argument('--session_id', default = 2, type=int)
    parser.add_argument('--data_root', default='./data/comb_dataset_v3', type=str)

    args = parser.parse_args()
    
    #if args.retrain:
    #    if args.base_model_path is None:
    #        print "Please provide a base model path."
    #        sys.exit()
    #    else:
    #        main(args)
    #else:
    died_num = 0
    while True:
        main(args)
        print(f'宕机{died_num}次')
        died_num += 1
