#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Tensorlayer version == 1.11.1
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, MaxPooling2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add, LeakyReLU, Lambda
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.activations import sigmoid
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
import cv2
from keras.applications.vgg19 import preprocess_input


# In[2]:


'''
Use LR rate as 0.0001 first then 0.00001
'''


# In[3]:


batch_size = 16
lr_rate = 0.00001
ni = np.sqrt(batch_size)
hr_shape = (224,224,3)
lr_shape = (112,112,3)


# In[4]:


def SubpixelConv2D(name, scale=2):
    """
    Keras layer to do subpixel convolution.
    NOTE: Tensorflow backend only. Uses tf.depth_to_space

    :param scale: upsampling scale compared to input_shape. Default=2
    :return:
    """

    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                None if input_shape[1] is None else input_shape[1] * scale,
                None if input_shape[2] is None else input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)

    return Lambda(subpixel, output_shape=subpixel_shape, name=name)


# In[5]:


def PSNR(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, see https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    The equation is:
    PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)

    Since input is scaled from -1 to 1, MAX_I = 1, and thus 20 * log10(1) = 0. Only the last part of the equation is therefore neccesary.
    """
    return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)

def discAcc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))


# In[6]:


def Generator(upscaling_factor):
    residual_blocks = 16
    def residual_block(_input):
        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(_input)
        x = BatchNormalization(momentum=0.8)(x)
        x = PReLU(shared_axes=[1,2])(x)            
        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Add()([x, _input])
        return x

    def upsample(x, number):
        x = Conv2D(256, kernel_size=3, strides=1, padding='same', name='upSampleConv2d_'+str(number))(x)
        x = SubpixelConv2D('upSampleSubPixel_'+str(number), 2)(x)
        x = PReLU(shared_axes=[1,2], name='upSamplePReLU_'+str(number))(x)
        return x

    # Input low resolution image
    lr_input = Input(shape=(None, None, 3))

    # Pre-residual
    x_start = Conv2D(64, kernel_size=9, strides=1, padding='same')(lr_input)
    x_start = PReLU(shared_axes=[1,2])(x_start)

    # Residual blocks
    r = residual_block(x_start)
    for _ in range(residual_blocks - 1):
        r = residual_block(r)

    # Post-residual block
    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
    x = BatchNormalization(momentum=0.8)(x)
    x = Add()([x, x_start])

    # Upsampling depending on factor
    x = upsample(x, 1)
    if upscaling_factor > 2:
        x = upsample(x, 2)
    if upscaling_factor > 4:
        x = upsample(x, 3)

    # Generate high resolution output
    # tanh activation, see: 
    # https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b
    hr_output = Conv2D(3, kernel_size=9, strides=1, padding='same', activation='tanh')(x)
    
    return Model(inputs=lr_input, outputs=hr_output) 

gen = Generator(2)
gen.compile(loss='mse',
           optimizer=Adam(lr_rate, 0.9),
           metrics=['mse',PSNR])


# In[7]:


def Discriminator(hr_shape,is_train=True):
    inputLayer = Input(hr_shape, name='in')
    n = Conv2D(64, kernel_size=4, strides=2, padding='same', name='n64s1/c')(inputLayer)
    n = LeakyReLU(alpha=0.2)(n)
    
    n = Conv2D(128, kernel_size=4, strides=2, padding='same', name='n64s2/c')(n)
    n = BatchNormalization()(n)
    n = LeakyReLU(alpha=0.2)(n)
    
    n = Conv2D(256, kernel_size=4, strides=2, padding='same', name='n128s1/c')(n)
    n = BatchNormalization()(n)
    n = LeakyReLU(alpha=0.2)(n)
    
    n = Conv2D(64*8, kernel_size=4, strides=2, padding='same', name='n128s2/c')(n)
    n = BatchNormalization()(n)
    n = LeakyReLU(alpha=0.2)(n)
    
    n = Conv2D(64*16, kernel_size=4, strides=2, padding='same', name='n256s1/c')(n)
    n = BatchNormalization()(n)
    n = LeakyReLU(alpha=0.2)(n)
    
    n = Conv2D(64*32, kernel_size=4, strides=2, padding='same', name='n256s2/c')(n)
    n = BatchNormalization()(n)
    n = LeakyReLU(alpha=0.2)(n)
    
    n = Conv2D(64*16, kernel_size=1, strides=1, padding='same', name='n512s1/c')(n)
    n = BatchNormalization()(n)
    n = LeakyReLU(alpha=0.2)(n)
    
    n = Conv2D(64*8, kernel_size=1, strides=1, padding='same', name='n512s2/c')(n)
    n = BatchNormalization()(n)
    n = LeakyReLU(alpha=0.2)(n)
    
    nn = Conv2D(64*2, kernel_size=1, strides=1, padding='same')(n)
    nn = BatchNormalization()(nn)
    nn = LeakyReLU(alpha=0.2)(nn)
    
    nn = Conv2D(64*2, kernel_size=3, strides=1, padding='same')(nn)
    nn = BatchNormalization()(nn)
    nn = LeakyReLU(alpha=0.2)(nn)
        
    nn = Conv2D(64*8, kernel_size=3, strides=1, padding='same')(nn)
    nn = BatchNormalization()(nn)
    nn = LeakyReLU(alpha=0.2)(nn)
    
    n = Add(name='res/add3')([n, nn])
    n = LeakyReLU(alpha=0.2)(n)
    
    n = Flatten()(n)
    n = Dense(1,activation='sigmoid')(n)
    
    return Model(inputLayer, n)

disc = Discriminator(hr_shape)
disc.compile(loss='binary_crossentropy',
            optimizer=Adam(lr_rate, 0.9),
            metrics=[discAcc])


# In[8]:


def preprocess_vgg(x):
    """Take a HR image [-1, 1], convert to [0, 255], then to input for VGG network"""
    if isinstance(x, np.ndarray):
        return preprocess_input((x+1)*127.5)
    else:
        return Lambda(lambda x: preprocess_input(tf.add(x, 1) * 127.5))(x) 


# In[9]:


def build_vgg(hr_shape):
    """
    Load pre-trained VGG weights from keras applications
    Extract features to be used in loss function from last conv layer, see architecture at:
    https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
    """

    # Input image to extract features from
    img = Input(hr_shape)

    # Get the vgg network. Extract features from last conv layer
    vgg = VGG19(weights="imagenet")
    vgg.outputs = [vgg.layers[20].output]

    # Create model and compile
    model = Model(inputs=img, outputs=vgg(img))
    model.trainable = False
    return model

vgg = build_vgg(hr_shape)
vgg.compile(
    loss='mse',
    optimizer=Adam(lr_rate, 0.9),
    metrics=['accuracy'])


# In[10]:


def build_srgan(lr_shape):
    """Create the combined SRGAN network"""

    # Input LR images
    img_lr = Input(lr_shape)

    # Create a high resolution image from the low resolution one
    generated_hr = gen(img_lr)
    generated_features = vgg(
        preprocess_vgg(generated_hr)
    )

    # In the combined model we only train the generator
    disc.trainable = False

    # Determine whether the generator HR images are OK
    generated_check = disc(generated_hr)

    # Create sensible names for outputs in logs
    generated_features = Lambda(lambda x: x, name='Content')(generated_features)
    generated_check = Lambda(lambda x: x, name='Adversarial')(generated_check)

    # Create model and compile
    # Using binary_crossentropy with reversed label, to get proper loss, see:
    # https://danieltakeshi.github.io/2017/03/05/understanding-generative-adversarial-networks/
    model = Model(inputs=img_lr, outputs=[generated_check, generated_features])        
    return model

srgan = build_srgan(lr_shape)
srgan.compile(loss=['binary_crossentropy','mse'],
             loss_weights=[1e-3, 0.006],
             optimizer=Adam(lr_rate, 0.9))


# In[11]:


path = "./mainDataset"
dataset_name = "train"
# for the VGG feature out true labels
ni = np.sqrt(batch_size)
#disc_patch = (14, 14, 1)


# In[12]:


def readImage(x,p):
    img = cv2.imread(p+x,cv2.IMREAD_COLOR)
    return img


# In[13]:


train_hr_img_list = sorted(tl.files.load_file_list(path=path+'/%s/' % (dataset_name), regx='.*.png', printable=False))
#train_hr_imgs = tl.vis.read_images(train_hr_img_list[0:100], path=path+'/%s/' % (dataset_name), n_threads=32)
train_hr_imgs = tl.prepro.threading_data(train_hr_img_list,fn=readImage,p=path+'/%s/' % (dataset_name))


# In[14]:


train_hr_imgs_clean = []
for i,img in enumerate(train_hr_imgs):
    if img is not None:
        train_hr_imgs_clean.append(img)
train_hr_imgs = np.array(train_hr_imgs_clean)


# In[15]:


steps = len(train_hr_imgs)//batch_size


# In[16]:


def scaleHR(x):
    x = x / (255. / 2.)
    x = x - 1.
    return x

def scaleLR(x):
    x = cv2.GaussianBlur(x,(3,3),0)
    x = x / 255.
    return x
    
def datagen(dev_hr_imgs,batchSize,is_testing=False):
    while(True):
        imgs_hr=[]
        imgs_lr=[]
        imgs = np.random.choice(dev_hr_imgs,batchSize)
        img_hr = tl.prepro.threading_data(imgs, fn=crop, wrg=224, hrg=224, is_random=True)
        #img_hr = tl.prepro.threading_data(imgs, fn=imresize,size=[224, 224], interp='bicubic', mode=None)
        img_lr = tl.prepro.threading_data(img_hr, fn=imresize,size=[112, 112], interp='bicubic', mode=None)
        
        imgs_hr = tl.prepro.threading_data(img_hr,fn=scaleHR)
        imgs_lr = tl.prepro.threading_data(img_lr,fn=scaleLR)
        
        yield imgs_hr, imgs_lr


# In[17]:


datagenObj = datagen(train_hr_imgs,batch_size)


# In[18]:


sample_hr,sample_lr = next(datagenObj)


# In[19]:


tl.vis.save_images(sample_hr, [int(ni), int(ni)],'images/'+dataset_name+'/sample_hr.png')
tl.vis.save_images(sample_lr, [int(ni), int(ni)],'images/'+dataset_name+'/sample_lr.png')


# In[23]:


'''
tensorboard = TensorBoard(
  log_dir='log/srgan_FAST/SRResNet/run1',
  histogram_freq=0,
  batch_size=batch_size,
  write_graph=True,
  write_grads=True
)
tensorboard.set_model(srgan)
'''


# In[ ]:


'''
for epoch in range(100):
    print("Epoch:"+str(epoch))
    for step in tqdm_notebook(range(0,steps)):
        imgs_hr, imgs_lr = next(datagenObj)

        # Train the generators
        g_loss = gen.train_on_batch(imgs_lr,imgs_hr)
    out = gen.predict(sample_lr)
    tl.vis.save_images(out, [int(ni), int(ni)],'images/'+dataset_name+'/train.png')
    tensorboard.on_epoch_end(epoch, {"g_loss": g_loss[2]})
    if(epoch % 10 == 0):
        out = gen.predict(sample_lr)
        tl.vis.save_images(out, [int(ni), int(ni)],'images/'+dataset_name+'/trainSRResNetFAST_%d.png' % int(epoch))
'''


# In[ ]:


'''
gen.save_weights("checkpoints/GenInitFAST.h5")
'''


# In[19]:


#gen.load_weights("checkpoints/GenInitFAST.h5")


# In[20]:


'''
modelcheckpoint = ModelCheckpoint(
    filepath = "./checkpoints/genFASTMC.h5",
    monitor='g_loss',
    verbose=0,
    mode="auto",
    save_best_only=True
)
modelcheckpoint.set_model(gen)
discMC = ModelCheckpoint(
    filepath = "./checkpoints/discFASTMC.h5",
    monitor='g_loss',
    verbose=0,
    mode="auto",
    save_best_only=True
)
discMC.set_model(disc)
'''


# In[21]:


'''
tensorboard = TensorBoard(
  log_dir='log/srgan_FAST/SRGAN_main/run1',
  histogram_freq=0,
  batch_size=batch_size,
  write_graph=True,
  write_grads=True
)
tensorboard.set_model(srgan)
'''


# In[ ]:


'''
for epoch in range(150):
    print("Epoch:"+str(epoch))
    for step in tqdm_notebook(range(0,steps)):
        imgs_hr, imgs_lr = next(datagenObj)
        # From low res. image generate high res. version
        fake_hr = gen.predict(imgs_lr)
        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))
        if step % 2 == 0:
            real_loss = disc.train_on_batch(imgs_hr, valid)
            fake_loss = disc.train_on_batch(fake_hr, fake)
            discriminator_loss = 0.5 * np.add(real_loss, fake_loss)
        else:
            features_hr = vgg.predict(preprocess_vgg(imgs_hr))
            generator_loss = srgan.train_on_batch(imgs_lr, [valid, features_hr]) 
        
    mc_step = epoch*steps + steps
    modelcheckpoint.on_epoch_end(mc_step,{"g_loss":generator_loss[1]})  
    discMC.on_epoch_end(mc_step,{"g_loss":generator_loss[1]})
    out = gen.predict(sample_lr)
    tl.vis.save_images(out, [int(ni), int(ni)],'images/'+dataset_name+'/train.png')
    if(epoch % 10 == 0):
        out = gen.predict(sample_lr)
        tl.vis.save_images(out, [int(ni), int(ni)],'images/'+dataset_name+'/FASTx2_%d.png' % int(epoch))
    tensorboard.on_epoch_end(epoch, {"d_loss": discriminator_loss[0],"d_acc":discriminator_loss[1],"g_binary_crossentropy":generator_loss[0], "g_vgg_loss":generator_loss[1]})
srgan.save_weights("checkpoints/srgan.h5")
tensorboard.on_train_end(None)
'''


# In[ ]:


'''
gen.save_weights("checkpoints/backupGenFAST.h5")
disc.save_weights("checkpoints/backupDiscFast.h5")
'''


# In[20]:


gen.load_weights("checkpoints/genFASTMC.h5")
disc.load_weights("checkpoints/discFASTMC.h5")


# In[ ]:


modelcheckpoint = ModelCheckpoint(
    filepath = "./checkpoints/genFASTMC_soft.h5",
    monitor='g_loss',
    verbose=0,
    mode="auto",
    save_best_only=True
)
modelcheckpoint.set_model(gen)
discMC = ModelCheckpoint(
    filepath = "./checkpoints/discFASTMC_soft.h5",
    monitor='g_loss',
    verbose=0,
    mode="auto",
    save_best_only=True
)
discMC.set_model(disc)


# In[ ]:


tensorboard = TensorBoard(
  log_dir='log/srgan_FAST/SRGAN_SOFT/run3',
  histogram_freq=0,
  batch_size=batch_size,
  write_graph=True,
  write_grads=True
)
tensorboard.set_model(srgan)


# In[ ]:


for epoch in range(150):
    print("Epoch:"+str(epoch))
    for step in range(0,steps):
        imgs_hr, imgs_lr = next(datagenObj)
        # From low res. image generate high res. version
        fake_hr = gen.predict(imgs_lr)
        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))
        if step % 2 == 0:
            real_loss = disc.train_on_batch(imgs_hr, valid)
            fake_loss = disc.train_on_batch(fake_hr, fake)
            discriminator_loss = 0.5 * np.add(real_loss, fake_loss)
        else:
            features_hr = vgg.predict(preprocess_vgg(imgs_hr))
            generator_loss = srgan.train_on_batch(imgs_lr, [valid, features_hr]) 
        
    mc_step = epoch*steps + steps
    modelcheckpoint.on_epoch_end(mc_step,{"g_loss":generator_loss[1]})  
    discMC.on_epoch_end(mc_step,{"g_loss":generator_loss[1]})
    out = gen.predict(sample_lr)
    tl.vis.save_images(out, [int(ni), int(ni)],'images/'+dataset_name+'/train.png')
    if(epoch % 10 == 0):
        out = gen.predict(sample_lr)
        tl.vis.save_images(out, [int(ni), int(ni)],'images/'+dataset_name+'/FASTx2_SOFT_%d.png' % int(epoch))
    tensorboard.on_epoch_end(epoch, {"d_loss": discriminator_loss[0],"d_acc":discriminator_loss[1],"g_binary_crossentropy":generator_loss[0], "g_vgg_loss":generator_loss[1]})
srgan.save_weights("checkpoints/srgan.h5")
tensorboard.on_train_end(None)


# In[ ]:


gen.save_weights("checkpoints/backupGenFASTSOFT.h5")
disc.save_weights("checkpoints/backupDiscFastSOFT.h5")

