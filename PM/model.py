from __future__ import division
import os
import time
import Queue
import threading
from PIL import Image
from cStringIO import StringIO
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, input_height=64, input_width=64, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=6, z_dim=256, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim
    self.is_train = tf.placeholder(tf.bool, [])

    # batch normalization : deals with poor initialization helps gradient flow
    with tf.variable_scope('GAN'):
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')




        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name='g_bn4')
        self.g_bn5 = batch_norm(name='g_bn5')
        self.g_bn6 = batch_norm(name='g_bn6')



        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir

        _,_,_,self.data,_,_=np.load('../'+self.dataset_name)

        #self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
        #self.c_dim = imread(self.data[0]).shape[-1]
        self.c_dim=3

        self.grayscale = (self.c_dim == 1)

        self.build_model()
  def get_gen(self,_z, _y):
    with tf.variable_scope('GAN') as scope:
      scope.reuse_variables()
      return self.generator(_z,_y)
  def get_dis(self,_z, _y):
    with tf.variable_scope('GAN') as scope:
      scope.reuse_variables()
      return self.discriminator(_z,_y) 
  def build_model(self):
    self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')
    self.sample_inputs = tf.placeholder(
      tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

    inputs = self.inputs
    sample_inputs = self.sample_inputs

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

 
    self.G = self.generator(self.z, self.y)
    self.D = \
        self.discriminator(inputs, self.y, reuse=False)

    self.sampler = self.sampler(self.z, self.y)
    self.D_ = \
        self.discriminator(self.G, self.y, reuse=True)
  

    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)



    self.d_loss_real = tf.reduce_mean(tf.square(self.D-1))
    self.d_loss_fake = tf.reduce_mean(tf.square(self.D_))
    self.g_loss = tf.reduce_mean(tf.square(self.D_-1))

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                          
    self.d_loss = (self.d_loss_real + self.d_loss_fake)/2

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver([k for k in tf.global_variables() if k.name.startswith('GAN')])

  def train(self, config):
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1, beta2=0.99) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1, beta2=0.99) \
              .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.g_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = SummaryWriter("./logs", self.sess.graph)

    sample_z = np.random.normal(size=(self.sample_num , self.z_dim))
    #sample_z = rand_z/np.linalg.norm(rand_z, axis=1, keepdims=True)
    
    def load_batch(q):
      imgs=np.zeros([self.batch_size,64,64,self.c_dim])
      labels=np.zeros([self.batch_size,self.y_dim])
      itemnum=len(self.data)
      while True:
        for i in range(self.batch_size):
          idx = np.random.randint(itemnum)
          jpg=np.asarray(Image.open(StringIO(self.data[idx]['imgs'])).convert('RGB').resize((64,64)))      
          jpg=(jpg-127.5)/127.5
          y=self.data[idx]['c']
          imgs[i],labels[i]=jpg,y
        q.put((imgs,labels))
    q=Queue.Queue(maxsize=5)
        
    for i in range(1):
      t = threading.Thread(target=load_batch,args=[q])
      t.start()
    
    
    
    
    if config.dataset == 'mnist':
      sample_inputs = self.data_X[0:self.sample_num]
      sample_labels = self.data_y[0:self.sample_num]
    else:
      sample_inputs, sample_labels = q.get()
      
  
    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
         
      #self.data = glob(os.path.join(
      #  "./data", config.dataset, self.input_fname_pattern))
      batch_idxs = min(len(self.data), config.train_size) // config.batch_size

      for idx in xrange(0, batch_idxs):
        if config.dataset == 'mnist':
          batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]
        else:
          batch_images,batch_labels=q.get()
          
        batch_z = np.random.normal(size=(config.batch_size, self.z_dim))
        #batch_z = batch_z/np.linalg.norm(batch_z,axis=1,keepdims=True)

        if True:
          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={ 
              self.inputs: batch_images,
              self.z: batch_z,
              self.y:batch_labels,
              self.is_train: True
            })
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={
              self.z: batch_z, 
              self.y:batch_labels,
              self.is_train: True
            })
          
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          #_, summary_str = self.sess.run([g_optim, self.g_sum],
          #  feed_dict={ self.z: batch_z, self.y:batch_labels })
          #self.writer.add_summary(summary_str, counter)
          
          errD_fake = self.d_loss_fake.eval({
              self.z: batch_z, 
              self.y:batch_labels,
              self.is_train: False
          })
          errD_real = self.d_loss_real.eval({
              self.inputs: batch_images,
              self.y:batch_labels,
              self.is_train: False
          })
          errG = self.g_loss.eval({
              self.z: batch_z,
              self.y: batch_labels,
              self.is_train: False
          })
        

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG))

        if np.mod(counter, 100) == 1:
          if config.dataset == 'mnist':
            samples, d_loss, g_loss = self.sess.run(
              [self.sampler, self.d_loss, self.g_loss],
              feed_dict={
                  self.z: sample_z,
                  self.inputs: sample_inputs,
                  self.y:sample_labels,
                  self.is_train: False
              }
            )
            manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
            manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
            save_images(samples, [manifold_h, manifold_w],
                  './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
          else:
            try:
              samples, d_loss, g_loss = self.sess.run(
                [self.sampler, self.d_loss, self.g_loss],
                feed_dict={
                    self.z: sample_z,
                    self.inputs: sample_inputs,
                    self.y:sample_labels,
                    self.is_train: False
                },
              )
              manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
              manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
              save_images(samples, [manifold_h, manifold_w],
                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
              print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
            except:
              print("one pic error!...")

        if np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)

  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()            
        yb = tf.reshape(y, [int(y.get_shape()[0]), 1, 1, self.y_dim])
        x = conv_cond_concat(image, yb)    
        h0 = lrelu(conv2d(x, 64, name='d_h0_conv'))
        h0 = conv_cond_concat(h0, yb)
        h1 = lrelu(self.d_bn1(conv2d(h0, 128, name='d_h1_conv'),is_train=self.is_train))
        h1 = conv_cond_concat(h1, yb)
        h2 = lrelu(self.d_bn2(conv2d(h1, 256, name='d_h2_conv'),is_train=self.is_train))
        h2 = conv_cond_concat(h2, yb)
        h3 = lrelu(self.d_bn3(conv2d(h2, 512, name='d_h3_conv'),is_train=self.is_train))
        #h3 = conv_cond_concat(h3, yb)
        #h4 = lrelu(self.d_bn4(linear(tf.reshape(h3, [int(h3.get_shape()[0]), -1]), 1024, 'd_h3_lin')))
        #h4 = tf.concat([h4, y],1)
        #h5 = linear(tf.reshape(h4, [int(h4.get_shape()[0]), -1]), 1, 'd_h4_lin')
        h4=linear(tf.reshape(h3, [int(h3.get_shape()[0]), -1]), 1, 'd_h4_lin')

  
        
        return h4

  def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
        
        z = tf.concat([z, y], 1)
        self.z_, self.h0_w, self.h0_b = linear(z, 4*4*256, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(self.z_, [-1, 4, 4, 256])
        h0 = lrelu(self.g_bn0(self.h0,is_train=self.is_train))

        self.h1, self.h1_w, self.h1_b = deconv2d(h0, 
                                                 [self.batch_size, 8, 8, 256], name='g_h1', with_w=True)
        h1 = lrelu(self.g_bn1(self.h1,is_train=self.is_train))
        
        self.h2, self.h2_w, self.h2_b = deconv2d(h1, 
                                                 [self.batch_size, 8, 8, 256], stride=1, name='g_h2', with_w=True)
        h2 = lrelu(self.g_bn2(self.h2,is_train=self.is_train))

        h3, self.h3_w, self.h3_b = deconv2d(h2,
                                            [self.batch_size, 16, 16, 256], name='g_h3', with_w=True)
        h3 = lrelu(self.g_bn3(h3,is_train=self.is_train))

        h4, self.h4_w, self.h4_b = deconv2d(h3,
                                            [self.batch_size, 16, 16, 256], stride=1, name='g_h4', with_w=True)
        h4 = lrelu(self.g_bn4(h4,is_train=self.is_train))

        h5, self.h5_w, self.h5_b = deconv2d(h4,
                                            [self.batch_size, 32, 32, 128], name='g_h5', with_w=True)
        h5 = lrelu(self.g_bn5(h5,is_train=self.is_train))
        
        h6, self.h6_w, self.h6_b = deconv2d(h5,
                                            [self.batch_size, 64, 64, 64], name='g_h6', with_w=True)
        h6 = lrelu(self.g_bn6(h6,is_train=self.is_train))
        
        h7, self.h7_w, self.h7_b = deconv2d(h6,
                                            [self.batch_size, 64, 64, 3], stride=1, name='g_h7', with_w=True)

        return tf.nn.tanh(h7)
    

  def sampler(self, z, y=None):
    with tf.variable_scope("generator") as scope:
        scope.reuse_variables()

        z = tf.concat([z, y], 1)
        self.z_, self.h0_w, self.h0_b = linear(z, 4*4*256, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(self.z_, [-1, 4, 4, 256])
        h0 = lrelu(self.g_bn0(self.h0,is_train=self.is_train))

        self.h1, self.h1_w, self.h1_b = deconv2d(h0, 
                                                 [self.batch_size, 8, 8, 256], name='g_h1', with_w=True)
        h1 = lrelu(self.g_bn1(self.h1,is_train=self.is_train))
        
        self.h2, self.h2_w, self.h2_b = deconv2d(h1, 
                                                 [self.batch_size, 8, 8, 256], stride=1, name='g_h2', with_w=True)
        h2 = lrelu(self.g_bn2(self.h2,is_train=self.is_train))

        h3, self.h3_w, self.h3_b = deconv2d(h2,
                                            [self.batch_size, 16, 16, 256], name='g_h3', with_w=True)
        h3 = lrelu(self.g_bn3(h3,is_train=self.is_train))

        h4, self.h4_w, self.h4_b = deconv2d(h3,
                                            [self.batch_size, 16, 16, 256], stride=1, name='g_h4', with_w=True)
        h4 = lrelu(self.g_bn4(h4,is_train=self.is_train))

        h5, self.h5_w, self.h5_b = deconv2d(h4,
                                            [self.batch_size, 32, 32, 128], name='g_h5', with_w=True)
        h5 = lrelu(self.g_bn5(h5,is_train=self.is_train))
        
        h6, self.h6_w, self.h6_b = deconv2d(h5,
                                            [self.batch_size, 64, 64, 64], name='g_h6', with_w=True)
        h6 = lrelu(self.g_bn6(h6,is_train=self.is_train))
        
        h7, self.h7_w, self.h7_b = deconv2d(h6,
                                            [self.batch_size, 64, 64, 3], stride=1, name='g_h7', with_w=True)

        return tf.nn.tanh(h7)


  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
