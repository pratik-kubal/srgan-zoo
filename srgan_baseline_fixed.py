#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Tensorlayer version == 1.11.1
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, MaxPooling2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add, LeakyReLU, Lambda
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D,Conv2DTranspose
from keras.applications import VGG19
from keras.models import Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import pydot
from keras.callbacks import TensorBoard, ModelCheckpoint
from glob import glob
from tqdm import tqdm
import scipy
import imageio
from tensorlayer.prepro import *
import tensorlayer as tl
from keras import initializers
from keras import backend as K
import tensorflow as tf
import h5py


# In[2]:


batch_size = 16
optimizer = Adam(lr=0.001, beta_1=0.9)
ni = np.sqrt(batch_size)
hr_shape = (224,224,3)
lr_shape = (56,56,3)


# In[3]:


def SubpixelConv2D(input_shape, scale=4):
    """
    Keras layer to do subpixel convolution.
    NOTE: Tensorflow backend only. Uses tf.depth_to_space
    Ref:
        [1] Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
            Shi et Al.
            https://arxiv.org/abs/1609.05158
    :param input_shape: tensor shape, (batch, height, width, channel)
    :param scale: upsampling scale. Default=4
    :return:
    """
    # upsample using depth_to_space
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)


    return Lambda(subpixel, output_shape=subpixel_shape)


# In[4]:


def Generator(lr_shape,is_train=True):
    w_init = initializers.RandomNormal(stddev=0.02, seed=None)
    g_init = initializers.RandomNormal(mean=0.0,stddev=0.02, seed=None)
    lr_input = Input(shape=lr_shape,name="in")
    n = Conv2D(64, kernel_size=9, strides=1, padding='same', kernel_initializer=w_init, bias_initializer='zeros', name='n64s1/c')(lr_input)
    temp = n
    # Res blocks
    for i in range(16):
        nn = Conv2D(64, (3, 3), strides=1, padding='same', kernel_initializer=w_init, bias_initializer='zeros', name='n64s1/c1/%s' % i)(n)
        if is_train: 
            nn = BatchNormalization(gamma_initializer=g_init, name='n64s1/b1/%s' % i)(nn)
            nn = Activation('relu')(nn)
        nn = Conv2D(64, (3, 3), strides=1, padding='same', kernel_initializer=w_init, bias_initializer='zeros', name='n64s1/c2/%s' % i)(nn)
        if is_train: 
            nn = BatchNormalization(gamma_initializer=g_init, name='n64s1/b2/%s' % i)(nn)
        nn = Add(name='b_residual_add/%s' % i)([n, nn])
        n = nn

    n = Conv2D(64, (3, 3), strides=1, padding='same', kernel_initializer=w_init, bias_initializer='zeros', name='n64s1/c/m')(n)
    if is_train: n = BatchNormalization(gamma_initializer=g_init, name='n64s1/b/m')(n)
    n = Add(name='add3')([n, temp])
    # End of Res Blocks

    n = Conv2D(256, kernel_size=3, strides=1, padding='same',kernel_initializer=w_init, bias_initializer='zeros')(n)
    pixelShuffle1 = SubpixelConv2D((None, 56, 56, 256),scale=2)(n)
    if is_train: pixelShuffle1 = BatchNormalization(gamma_initializer=g_init)(pixelShuffle1)
    pixelShuffle1 = Activation('relu')(pixelShuffle1)

    pixelShuffle2 = Conv2D(256, kernel_size=3, strides=1, padding='same',kernel_initializer=w_init, bias_initializer='zeros')((pixelShuffle1))
    pixelShuffle2 = SubpixelConv2D((None, 112, 112, 256),scale=2)(pixelShuffle2)
    if is_train: pixelShuffle2 = BatchNormalization(gamma_initializer=g_init)(pixelShuffle2)
    pixelShuffle2 = Activation('relu')(pixelShuffle2)

    out = Conv2D(3, (9, 9),strides=1, activation='tanh', padding='same', kernel_initializer=w_init, name='out')(pixelShuffle2)
    return Model(lr_input,out)

SRGAN_gen = Generator(lr_shape)
SRGAN_gen.summary()


# In[5]:


def Discriminator(hr_shape,is_train=True):
    w_init = initializers.RandomNormal(stddev=0.02, seed=None)
    gamma_init = initializers.RandomNormal(mean=1,stddev=0.02, seed=None)
    inputLayer = Input(hr_shape, name='in')
    n = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer=w_init, bias_initializer='zeros', name='n64s1/c')(inputLayer)
    n = LeakyReLU(alpha=0.2)(n)
    
    n = Conv2D(64, kernel_size=3, strides=2, padding='same', kernel_initializer=w_init, bias_initializer='zeros', name='n64s2/c')(n)
    if is_train: n = BatchNormalization(gamma_initializer=gamma_init)(n)
    n = LeakyReLU(alpha=0.2)(n)
    
    n = Conv2D(64*2, kernel_size=3, strides=1, padding='same', kernel_initializer=w_init, bias_initializer='zeros', name='n128s1/c')(n)
    if is_train: n = BatchNormalization(gamma_initializer=gamma_init)(n)
    n = LeakyReLU(alpha=0.2)(n)
    
    n = Conv2D(64*2, kernel_size=3, strides=2, padding='same', kernel_initializer=w_init, bias_initializer='zeros', name='n128s2/c')(n)
    if is_train: n = BatchNormalization(gamma_initializer=gamma_init)(n)
    n = LeakyReLU(alpha=0.2)(n)
    
    n = Conv2D(64*4, kernel_size=3, strides=1, padding='same', kernel_initializer=w_init, bias_initializer='zeros', name='n256s1/c')(n)
    if is_train: n = BatchNormalization(gamma_initializer=gamma_init)(n)
    n = LeakyReLU(alpha=0.2)(n)
    
    n = Conv2D(64*4, kernel_size=3, strides=2, padding='same', kernel_initializer=w_init, bias_initializer='zeros', name='n256s2/c')(n)
    if is_train: n = BatchNormalization(gamma_initializer=gamma_init)(n)
    n = LeakyReLU(alpha=0.2)(n)
    
    n = Conv2D(64*8, kernel_size=3, strides=1, padding='same', kernel_initializer=w_init, bias_initializer='zeros', name='n512s1/c')(n)
    if is_train: n = BatchNormalization(gamma_initializer=gamma_init)(n)
    n = LeakyReLU(alpha=0.2)(n)
    
    n = Conv2D(64*8, kernel_size=3, strides=2, padding='same', kernel_initializer=w_init, bias_initializer='zeros', name='n512s2/c')(n)
    if is_train: n = BatchNormalization(gamma_initializer=gamma_init)(n)
    n = LeakyReLU(alpha=0.2)(n)
    
    n = Flatten()(n)
    n = Dense(1024)(n)
    n = LeakyReLU(alpha=0.2)(n)
    n = Dense(1,activation='sigmoid')(n)
    
    return Model(inputLayer, n)

SRGAN_disc = Discriminator(hr_shape)
SRGAN_disc.trainable = False
SRGAN_disc.compile(loss='mean_squared_error',
             optimizer=optimizer,
             metrics=['binary_accuracy'])
SRGAN_disc.summary()


# In[6]:


def buildVGG(hr_shape):
    vggInput = Input(shape=hr_shape)
    vgg = VGG19(weights="imagenet",input_tensor = vggInput)
    
    return Model(vggInput,vgg.layers[15].output)

vggModel = buildVGG(hr_shape)
for l in vggModel.layers:
    l.trainable = False


# In[7]:


vggModel.summary()


# In[8]:


img_lr = Input(shape=lr_shape)
img_hr = Input(shape=hr_shape)
gen_hr = SRGAN_gen(img_lr)
validity = SRGAN_disc(gen_hr)
vgg_features = vggModel(gen_hr)
combined = Model(img_lr, [validity, vgg_features])
combined.compile(loss=['binary_crossentropy', 'mean_squared_error'],
                 loss_weights=[1e-3, 0.006],
                 optimizer=optimizer)
combined.summary()


# In[ ]:


# SVG(model_to_dot(combined, show_layer_names=True, show_shapes=True).create(prog='dot', format='svg'))


# In[9]:


path = "./mainDataset"
dataset_name = "train"
# for the VGG feature out true labels
disc_patch = (28, 28, 512)
ni = np.sqrt(batch_size)


# In[10]:


train_hr_img_list = sorted(tl.files.load_file_list(path=path+'/%s/' % (dataset_name), regx='.*.png', printable=False))
train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=path+'/%s/' % (dataset_name), n_threads=32)


# In[20]:


steps = len(train_hr_imgs)//batch_size
'''
IdealEpochs = 100000/steps
'''
epochs = 300


# In[14]:


def scale(x):
    x = x / (255. / 2.)
    x = x - 1.
    return x
    
def datagen(dev_hr_imgs,batchSize,is_testing=False):
    while(True):
        imgs_hr=[]
        imgs_lr=[]
        imgs = np.random.choice(dev_hr_imgs,batchSize)
        img_hr = tl.prepro.threading_data(imgs, fn=crop, wrg=224, hrg=224, is_random=True)
        img_lr = tl.prepro.threading_data(img_hr, fn=imresize,size=[56, 56], interp='bicubic', mode=None)
        
        imgs_hr = tl.prepro.threading_data(img_hr,fn=scale)
        imgs_lr = tl.prepro.threading_data(img_lr,fn=scale)
        
        yield imgs_hr, imgs_lr


# In[15]:


datagenObj = datagen(train_hr_imgs,batch_size)


# In[16]:


sample_hr,sample_lr = next(datagenObj)


# In[17]:


tl.vis.save_images(sample_hr, [int(ni), int(ni)],'images/'+dataset_name+'/sample_hr.png')
tl.vis.save_images(sample_lr, [int(ni), int(ni)],'images/'+dataset_name+'/sample_lr.png')


# In[ ]:


tensorboard = TensorBoard(
  log_dir='log/srgan_final/hardTrain/run3',
  histogram_freq=0,
  batch_size=batch_size,
  write_graph=True,
  write_grads=True
)
tensorboard.set_model(combined)


# In[19]:


# Training SRResNet
SRGAN_disc.trainable = False
for epoch in range(100):
    print("Epoch:"+str(epoch))
    for step in tqdm(range(0,steps)):
        imgs_hr, imgs_lr = next(datagenObj)
        # The generators want the discriminators to label the generated images as real
        valid = np.ones(batch_size)

        # Extract ground truth image features using pre-trained VGG19 model
        image_features = vggModel.predict(imgs_hr)

        # Train the generators
        g_loss = combined.train_on_batch(imgs_lr, [valid, image_features])
        
        
    out = SRGAN_gen.predict(sample_lr)
    tl.vis.save_images(out, [int(ni), int(ni)],'images/'+dataset_name+'/train.png')
    if(epoch % 10 == 0):
        out = SRGAN_gen.predict(sample_lr)
        tl.vis.save_images(out, [int(ni), int(ni)],'images/'+dataset_name+'/trainSRResNet_%d.png' % int(epoch))


# In[29]:


for epoch in range(epochs):
    print("Epoch:"+str(epoch))
    for step in tqdm(range(0,steps)):
        # Sample images and their conditioning counterparts
        imgs_hr, imgs_lr = next(datagenObj)
        # From low res. image generate high res. version
        fake_hr = SRGAN_gen.predict(imgs_lr)
        if step % 2 == 0:
            # ----------------------
            #  Train Discriminator
            # ----------------------
            
            valid = np.ones(batch_size)
            fake = np.zeros(batch_size)

            # Train the discriminators (original images = real / generated = Fake)
            SRGAN_disc.trainable = True
            d_loss_real = SRGAN_disc.train_on_batch(imgs_hr, valid)
            d_loss_fake = SRGAN_disc.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            SRGAN_disc.trainable = False
            
        else:
            # ------------------
            #  Train Generator
            # ------------------

            # The generators want the discriminators to label the generated images as real
            valid = np.ones(batch_size)

            # Extract ground truth image features using pre-trained VGG19 model
            image_features = vggModel.predict(imgs_hr)

            # Train the generators
            g_loss = combined.train_on_batch(imgs_lr, [valid, image_features])
        
        
    out = SRGAN_gen.predict(sample_lr)
    tl.vis.save_images(out, [int(ni), int(ni)],'images/'+dataset_name+'/train.png')
    if(epoch % 10 == 0):
        out = SRGAN_gen.predict(sample_lr)
        tl.vis.save_images(out, [int(ni), int(ni)],'images/'+dataset_name+'/HardTrain_%d.png' % int(epoch))
    tensorboard.on_epoch_end(epoch, {"d_mse": d_loss[0],"d_acc":d_loss[1],"g_loss":g_loss[0],"g_mse":g_loss[1]})
    SRGAN_gen.save_weights("./checkpoints/gen.h5")
    SRGAN_disc.save_weights("./checkpoints/disc.h5")
tensorboard.on_train_end(None)


# In[22]:


optimizer = Adam(lr=0.0001, beta_1=0.9)
epochs = 300


# In[27]:

SRGAN_disc.compile(loss='mean_squared_error',
             optimizer=optimizer,
             metrics=['binary_accuracy'])
combined.compile(loss=['binary_crossentropy', 'mean_squared_error'],
                 loss_weights=[1e-3, 0.006],
                 optimizer=optimizer)


# In[30]:


SRGAN_gen.load_weights("./checkpoints/gen.h5")
SRGAN_disc.load_weights("./checkpoints/disc.h5")


# In[ ]:


tensorboard = TensorBoard(
  log_dir='log/srgan_final/softTrain/run3',
  histogram_freq=0,
  batch_size=batch_size,
  write_graph=True,
  write_grads=True
)
tensorboard.set_model(combined)


# In[ ]:


for epoch in range(epochs):
    print("Epoch:"+str(epoch))
    for step in tqdm(range(0,steps)):
        # Sample images and their conditioning counterparts
        imgs_hr, imgs_lr = next(datagenObj)
        # From low res. image generate high res. version
        fake_hr = SRGAN_gen.predict(imgs_lr)
        if step % 2 == 0:
            # ----------------------
            #  Train Discriminator
            # ----------------------
            
            valid = np.ones(batch_size)
            fake = np.zeros(batch_size)

            # Train the discriminators (original images = real / generated = Fake)
            SRGAN_disc.trainable = True
            d_loss_real = SRGAN_disc.train_on_batch(imgs_hr, valid)
            d_loss_fake = SRGAN_disc.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            SRGAN_disc.trainable = False
            
        else:
            # ------------------
            #  Train Generator
            # ------------------

            # The generators want the discriminators to label the generated images as real
            valid = np.ones(batch_size)

            # Extract ground truth image features using pre-trained VGG19 model
            image_features = vggModel.predict(imgs_hr)

            # Train the generators
            g_loss = combined.train_on_batch(imgs_lr, [valid, image_features])
        
        
    out = SRGAN_gen.predict(sample_lr)
    tl.vis.save_images(out, [int(ni), int(ni)],'images/'+dataset_name+'/train.png')
    if(epoch % 10 == 0):
        out = SRGAN_gen.predict(sample_lr)
        tl.vis.save_images(out, [int(ni), int(ni)],'images/'+dataset_name+'/softTrain_%d.png' % int(epoch))
    tensorboard.on_epoch_end(epoch, {"d_mse": d_loss[0],"d_acc":d_loss[1],"g_loss":g_loss[0],"g_mse":g_loss[1]})
    SRGAN_gen.save_weights("./checkpoints/gen.h5")
    SRGAN_disc.save_weights("./checkpoints/disc.h5")
tensorboard.on_train_end(None)


# In[ ]:




