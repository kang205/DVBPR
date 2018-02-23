import sys
import math
import random
import time
from PIL import Image
import Queue
import numpy as np
import threading
from cStringIO import StringIO
import tensorflow as tf

dataset_name = 'AmazonFashion6ImgPartitioned.npy'

#Hyper-prameters
K = 100 # Latent dimensionality
lambda1 = 0.001 # Weight decay
lambda2 = 1.0 # Regularizer for theta_u
learning_rate = 1e-4
training_epoch = 20
batch_size = 128
dropout = 0.5 # Dropout, probability to keep units
numldprocess=4 # multi-threading for loading images


dataset = np.load('../'+dataset_name)

[user_train, user_validation, user_test, Item, usernum, itemnum] = dataset

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def avgpool2d(x, k=2):
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
       
# Create CNN model
def CNN(x,dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 224, 224, 3])


    conv1 = conv2d(x, Weights('wc1'), Biases('bc1'), strides=4)
    conv1 = tf.nn.relu(conv1)
    conv1 = maxpool2d(conv1, k=2)
    
    conv2 = conv2d(conv1, Weights('wc2'), Biases('bc2'))
    conv2 = tf.nn.relu(conv2)
    conv2 = maxpool2d(conv2, k=2)
    
    conv3 = conv2d(conv2, Weights('wc3'), Biases('bc3'))
    conv3 = tf.nn.relu(conv3)
    
    conv4 = conv2d(conv3, Weights('wc4'), Biases('bc4'))
    conv4 = tf.nn.relu(conv4)
    
    conv5 = conv2d(conv4, Weights('wc5'), Biases('bc5'))
    conv5 = tf.nn.relu(conv5)
    conv5 = maxpool2d(conv5, k=2)

    fc1 = tf.reshape(conv5, [-1,weights['wd1'][0]])
    fc1 = tf.add(tf.matmul(fc1, Weights('wd1')), Biases('bd1'))
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    
    fc2 = tf.add(tf.matmul(fc1, Weights('wd2')), Biases('bd2'))
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout)
    
    fc3 = tf.add(tf.matmul(fc2, Weights('wd3')), Biases('bd3'))
    
    return fc3

#define model
with tf.device('/gpu:0'):
    #training sample
    queueu = tf.placeholder(dtype=tf.int32,shape=[1])
    queuei = tf.placeholder(dtype=tf.int32,shape=[1])
    queuej = tf.placeholder(dtype=tf.int32,shape=[1])
    queueimage1 = tf.placeholder(dtype=tf.uint8,shape=[224,224,3])
    queueimage2 = tf.placeholder(dtype=tf.uint8,shape=[224,224,3])
    batch_train_queue = tf.FIFOQueue(batch_size*5, dtypes=[tf.int32,tf.int32,tf.int32,tf.uint8,tf.uint8], shapes=[[1],[1],[1],[224,224,3],[224,224,3]])
    batch_train_queue_op = batch_train_queue.enqueue([queueu,queuei,queuej,queueimage1,queueimage2]);
    u,i,j,image1,image2 = batch_train_queue.dequeue_many(batch_size)

    image_test=tf.placeholder(dtype=tf.uint8,shape=[batch_size,224,224,3])
    
    image1=(tf.to_float(image1)-127.5)/127.5
    image2=(tf.to_float(image2)-127.5)/127.5
    _image_test=(tf.to_float(image_test)-127.5)/127.5

    u=tf.reshape(u,shape=[batch_size])
    i=tf.reshape(i,shape=[batch_size])
    j=tf.reshape(j,shape=[batch_size])
    
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

    #siamese networks
    with tf.variable_scope("DVBPR") as scope:
        result1 = CNN(image1,dropout)
        scope.reuse_variables()
        result2 = CNN(image2,dropout)
        result_test = CNN(_image_test,1.0)
        nn_regularizers = sum(map(tf.nn.l2_loss,[Weights('wd1'), Weights('wd2'), Weights('wd3'), Weights('wc1'), Weights('wc2'), Weights('wc3'), Weights('wc4'), Weights('wc5')]))
        thetau = tf.Variable(tf.random_uniform([usernum,K],minval=0,maxval=1)/100)
   
    cost_train = tf.reduce_sum(tf.log(tf.sigmoid(tf.reduce_sum(tf.multiply(tf.gather(thetau,u),tf.subtract(result1,result2)),1,keep_dims=True))))
    regularizers = tf.nn.l2_loss(tf.gather(thetau,u))
    cost_train -= lambda1 * nn_regularizers + lambda2 * regularizers
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-cost_train)    
    
# Initializing the variables
init = tf.initialize_all_variables()

def AUC(train,test,U,I):
    ans=0
    cc=0
    for u in train:    
        i=test[u][0]['productid']
        T=np.dot(U[u,:],I.T)
        cc+=1
        M=set()      
        for item in train[u]:
            M.add(item['productid'])
        M.add(i)
            
        count=0
        tmpans=0
        #for j in xrange(itemnum):
        for j in random.sample(xrange(itemnum),100): #sample
            if j in M: continue
            if T[i]>T[j]: tmpans+=1
            count+=1
        tmpans/=float(count)
        ans+=tmpans
    ans/=float(cc)
    return ans

def Evaluation(step):
    print '...'
    U=sess.run(thetau)
    I=np.zeros([itemnum,K],dtype=np.float32)
    idx=np.array_split(range(itemnum),(itemnum+batch_size-1)/batch_size)
    
    input_images=np.zeros([batch_size,224,224,3],dtype=np.int8)
    for i in range(len(idx)):
        cc=0
        for j in idx[i]:
            input_images[cc]=np.uint8(np.asarray(Image.open(StringIO(Item[j]['imgs'])).convert('RGB').resize((224,224))))
            cc+=1
        I[idx[i][0]:(idx[i][-1]+1)]=sess.run(result_test,feed_dict={image_test:input_images})[:(idx[i][-1]-idx[i][0]+1)]
    print 'export finised!'
    np.save('UI_'+str(K)+'_'+str(step)+'.npy',[U,I])
    return AUC(user_train,user_validation,U,I), AUC(user_train,user_test,U,I)

def sample(user):
    u = random.randrange(usernum)
    numu = len(user[u])
    i = user[u][random.randrange(numu)]['productid']
    M=set()
    for item in user[u]:
        M.add(item['productid'])
    while True:
        j=random.randrange(itemnum)
        if (not j in M): break
    return (u,i,j)

def load_image_async():
    while True:
        (uuu,iii,jjj)=sample(user_train)
        jpg1=np.uint8(np.asarray(Image.open(StringIO(Item[iii]['imgs'])).convert('RGB').resize((224,224))))
        jpg2=np.uint8(np.asarray(Image.open(StringIO(Item[jjj]['imgs'])).convert('RGB').resize((224,224))))
        sess.run(batch_train_queue_op,feed_dict={queueu:np.asarray([uuu]),
                                                 queuei:np.asarray([iii]),
                                                 queuej:np.asarray([jjj]),
                                                 queueimage1:jpg1,queueimage2:jpg2,
                                                })
    
f=open('DVBPR.log','w')
config = tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
sess=tf.Session(config=config)
sess.run(init)

t=[0]*numldprocess
for i in range(numldprocess):
    t[i] = threading.Thread(target=load_image_async)
    t[i].daemon=True
    t[i].start()

oneiteration = 0
for item in user_train: oneiteration+=len(user_train[item])

step = 1
saver = tf.train.Saver([k for k in tf.global_variables() if k.name.startswith('DVBPR')])

epoch=0
while step * batch_size <= training_epoch*oneiteration+1:

    sess.run(optimizer, feed_dict={keep_prob: dropout})
    
    print 'Step#'+str(step)+' CNN update'

    if step*batch_size / oneiteration >epoch:
        epoch+=1
        saver.save(sess,'./DVBPR_auc_'+str(K)+'_'+str(step)+'.ckpt')
        auc_valid,auc_test=Evaluation(step)
        print 'Epoch #'+str(epoch)+':'+str(auc_test)+' '+str(auc_valid)+'\n'
        f.write('Epoch #'+str(epoch)+':'+str(auc_test)+' '+str(auc_valid)+'\n')
        f.flush()
    
    step += 1
print "Optimization Finished!"

