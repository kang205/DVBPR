import sys
import math
import json
import random
import time
from PIL import Image
import Queue
import scipy.misc

import numpy as np


dataset_name = '../AmazonFashion6ImgPartitioned.npy'
DVBPR_ckpt_path = '../DVBPR/DVBPR_auc_100_39728.ckpt'

dataset = np.load(dataset_name)



[user_train, user_validation, user_test, Item, usernum, itemnum] = dataset




import tensorflow as tf
K=D=100
dropout = 0.5 # Dropout, probability to keep units

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def avgpool2d(x, k=2):
    # AvgPool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


weights = {
    'wc1': [11, 11, 3, 64],
    'wc2': [5, 5, 64, 256],
    'wc3': [3, 3, 256, 256],
    'wc4': [3, 3, 256, 256],
    'wc5': [3, 3, 256, 256],    
    'wd1': [7*7*256, 4096],
    'wd2': [4096, 4096],
    'wd3': [4096, K],
}

biases = {
    'bc1': [64],
    'bc2': [256],
    'bc3': [256],
    'bc4': [256],
    'bc5': [256],
    'bd1': [4096],
    'bd2': [4096],
    'bd3': [K],
}



def Weights(name):
    return tf.get_variable(name,dtype=tf.float32,shape=weights[name],initializer=tf.contrib.layers.xavier_initializer())

def Biases(name):
    return tf.get_variable(name,dtype=tf.float32,initializer=tf.zeros(biases[name]))

                           
# Create model
def CNN(x,dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 224, 224, 3])

    # Convolution Layer
    conv1 = conv2d(x, Weights('wc1'), Biases('bc1'), strides=4)
    conv1 = tf.nn.relu(conv1)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    
    
    # Convolution Layer
    conv2 = conv2d(conv1, Weights('wc2'), Biases('bc2'))
    conv2 = tf.nn.relu(conv2)
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    
    # Convolution Layer
    conv3 = conv2d(conv2, Weights('wc3'), Biases('bc3'))
    conv3 = tf.nn.relu(conv3)
    
     # Convolution Layer
    conv4 = conv2d(conv3, Weights('wc4'), Biases('bc4'))
    conv4 = tf.nn.relu(conv4)
     # Convolution Layer
    conv5 = conv2d(conv4, Weights('wc5'), Biases('bc5'))
    conv5 = tf.nn.relu(conv5)
    # Max Pooling (down-sampling)
    conv5 = maxpool2d(conv5, k=2)

    

    # Fully connected layer
    # Reshape conv5 output to fit fully connected layer input
    fc1 = tf.reshape(conv5, [-1,weights['wd1'][0]])

    fc1 = tf.add(tf.matmul(fc1, Weights('wd1')), Biases('bd1'))
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    
    fc2 = tf.add(tf.matmul(fc1, Weights('wd2')), Biases('bd2'))
    fc2 = tf.nn.relu(fc2)
    # Apply Dropout
    fc2 = tf.nn.dropout(fc2, dropout)
    
    fc3 = tf.add(tf.matmul(fc2, Weights('wd3')), Biases('bd3'))
    
    return fc3


from cStringIO import StringIO
def loadimg(item):
    return np.round(np.array(Image.open(StringIO(Item[item]['imgs'])).convert('RGB').resize((224,224)),dtype=np.float64))


with tf.device('/gpu:0'):
    image_test=tf.placeholder(dtype=tf.uint8,shape=[64,224,224,3])
    _image_test=(tf.to_float(image_test)-127.5)/127.5
    
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

    with tf.variable_scope("DVBPR") as scope:
        result_test = CNN(_image_test,1.0)
        thetau = tf.Variable(tf.random_uniform([usernum,D],minval=0,maxval=1)/100)



# Initializing the variables
init = tf.initialize_all_variables()
config = tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
config.gpu_options.allow_growth = True

sess=tf.Session(config=config)


sess.run(init)




saver = tf.train.Saver([k for k in tf.global_variables() if k.name.startswith('DVBPR')])


saver.restore(sess,DVBPR_ckpt_path)



from model import DCGAN

dcgan = DCGAN(
          sess,
          input_width=64,
          input_height=64,
          output_width=64,
          output_height=64,
          batch_size=16,
          sample_num=16,
          y_dim=6,
          dataset_name='AmazonFashion6ImgPartitioned.npy',
          input_fname_pattern='*,jpg',
          crop='False',
          checkpoint_dir='../GAN/checkpoint',
          sample_dir='.')

dcgan.load('../GAN/checkpoint')


with tf.device('/gpu:0'):
    x=np.random.normal(0,0.5,size=[16,256])
    z=tf.Variable(x,name='input_code',dtype=tf.float32)
    y=tf.placeholder(dtype=tf.float32,shape=[16,6])
    gan_image=dcgan.get_gen(z, y)
    gan_rf=dcgan.get_dis(gan_image, y)
    
    image=tf.image.resize_nearest_neighbor(images=gan_image, size=[224,224], align_corners=None, name=None)
    
    with tf.variable_scope("DVBPR") as scope:
        scope.reuse_variables()
        result = CNN(image,1.0)

lamda=1.0
user=tf.placeholder(dtype=tf.int32,shape=[1])

with tf.variable_scope('opt'):
    obj=tf.reduce_mean(tf.matmul(result,tf.transpose(tf.gather(thetau,user))))-tf.reduce_mean(tf.square(gan_rf-1))*lamda
    optimizer = tf.train.AdamOptimizer(learning_rate=0.05).minimize(-obj,var_list=[z])
    idx=tf.reduce_sum(tf.matmul(result,tf.transpose(tf.gather(thetau,user))),1)

sess.run(tf.variables_initializer([k for k in tf.global_variables() if k.name.startswith('opt')]))


for cat in range(6):
    
    while True:
        _user=np.random.randint(usernum)
        if len(user_train[_user])>10 and Item[user_test[_user][0]['productid']]['c'][cat]==1: break
        
    

    nowy=np.zeros([16,6])
    for i in range(16): nowy[i,cat]=1

    
    
    x=np.random.normal(0,0.5,size=[16,256])
    sess.run(z.assign(x))
    
    IMG=[]
    P_SCORE=[]
    img,p_score=sess.run([gan_image,idx],feed_dict={dcgan.is_train:False,y:nowy,user:[_user]})
    IMG.append(img)
    P_SCORE.append(p_score)
    
    for i in range(10):
        objv,_=sess.run([idx,optimizer],feed_dict={dcgan.is_train:False,y:nowy,user:[_user]})
        img=sess.run(gan_image,feed_dict={dcgan.is_train:False,y:nowy,user:[_user]})
        IMG.append(img)
        P_SCORE.append(objv)
    
    
    for i in range(16):
        outimg=np.zeros([64,64,3])
        for j in range(11):
            outimg=np.concatenate([outimg,np.float64(np.round(IMG[j][i]*127.5+127.5))],axis=1)
        if i==0: _outimg=outimg
        else: _outimg=np.concatenate([_outimg,outimg],axis=0)
    scipy.misc.toimage(_outimg,cmin=0.0).save('gan_'+str(cat)+'.jpg')
    print np.array(P_SCORE).T

    
