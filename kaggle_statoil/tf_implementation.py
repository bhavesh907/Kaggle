# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 12:43:16 2018

@author: uesr
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import join as opj
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
import tensorflow as tf


import dataset1

train = pd.read_json("train.json")

data = dataset1.load_train("C:\\Users\\uesr\\Downloads\\kaggle_statoil\train\\train.json", 75, 2)

test = pd.read_json("test.json")


#Generate the training data
#Create 3 bands having HH, HV and avg of both
X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_train = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)


logs_path = "/tmp/statoil/1"

#this is testing
# Convolutional Layer 1.
filter_size1 = 3 
num_filters1 = 64

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3

# image dimensions (only squares for now)
img_size = 75

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info

num_classes = 2

# batch size
batch_size = 16

# validation split
validation_size = .2

# how long to wait after validation loss stops improving before terminating training
early_stopping = 4 # use None if you don't want to implement early stoping


def weights(shape):
    w = tf.truncated_normal(shape,stddev=0.01, dtype=np.float32)
    return tf.Variable(w)

def bias(shape):
    b = tf.zeros(shape)
    return tf.Variable(b)

def conv(input,              # The previous layer.
         num_input_channels, # Num. channels in prev. layer.
         filter_size,        # Width and height of each filter.
         num_filters,        # Number of filters.
         use_pooling=True):
    #1st convolution layer
     
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    
    with tf.name_scope('Weights'):
        
        weight = weights(shape=shape)
    
    conv1 = tf.nn.conv2d(input,
                         filter = weight,
                         strides = [1,1,1,1],
                         padding = 'VALID',
                         name = 'Conv1')
    
    relu1 = tf.nn.relu(conv1 + bias(num_filters))
    
    pool1 = tf.nn.max_pool(relu1,
                           ksize = [1,3,3,1],
                           strides = [1,2,2,1], 
                           padding='VALID',
                           name = 'Pool1')
    print (pool1.shape)
    
    drop1 = tf.nn.dropout(pool1,0.2)
    
    conv2 = tf.nn.conv2d(drop1,
                         filter = weights([3,3,64,128]),
                         strides = [1,1,1,1],
                         padding = 'VALID',
                         name = 'Conv2')
    
    relu2 = tf.nn.relu(conv2 + bias(128))
    
    pool2 = tf.nn.max_pool(relu2,
                           ksize = [1,2,2,1],
                           strides = [1,2,2,1], 
                           padding='VALID',
                           name = 'Pool2')
    print (pool2.shape)
    
    drop2 = tf.nn.dropout(pool2,0.2)
    
    conv3 = tf.nn.conv2d(drop2,
                         filter = weights([3,3,128,128]),
                         strides = [1,1,1,1],
                         padding = 'VALID',
                         name = 'Conv3')
    
    relu3 = tf.nn.relu(conv3 + bias(128))
    
    pool3 = tf.nn.max_pool(relu3,
                           ksize = [1,2,2,1],
                           strides = [1,2,2,1], 
                           padding='VALID',
                           name = 'Pool3')
    print (pool3.shape)
    
    drop3 = tf.nn.dropout(pool3,0.2)
    
    conv4 = tf.nn.conv2d(drop3,
                         filter = weights([3,3,128,64]),
                         strides = [1,1,1,1],
                         padding = 'VALID',
                         name = 'Conv4')
    
    relu4 = tf.nn.relu(conv4 + bias(64))
    
    pool4 = tf.nn.max_pool(relu4,
                           ksize = [1,2,2,1],
                           strides = [1,2,2,1], 
                           padding='VALID',
                           name = 'Pool4')
    print (pool4.shape)
    
    drop4 = tf.nn.dropout(pool4,0.2)
    
    h_flat = tf.reshape(drop4, [-1,256])
    
    print("shape of h_flat is",h_flat.shape)
    
    W_fc1 = weights([256,512])
    b_fc1 = bias([1,512])

    h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)
    
    W_fc2 = weights([512,256])
    b_fc2 = bias([1,256])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    
    W_fc3 = weights([256,1])
    b_fc3 = bias([1,1])

    h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
    
    print(h_fc3.shape)
    
    
    
    return h_fc3 ,weight

    
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, 1], name='y_true')

y_true_cls = tf.argmax(y_true, axis=1)


layer_conv1, weights_conv1 = conv(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

with tf.name_scope("softmax"):
    y_pred = tf.nn.softmax(layer_conv1)

y_pred_cls = tf.argmax(y_pred, axis=1)

with tf.name_scope('cross_entropy'):

    cross_entropy = -tf.reduce_sum(y_true * tf.log(y_pred), [1])
    
   '''cross_entropy =  tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,
                                                        labels=y_true)'''

cost = tf.reduce_mean(cross_entropy)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)
with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#tf.summary.scalar("cost", cross_entropy)
'''tf.summary.scalar("accuracy", accuracy)'''

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
#summary_op = tf.summary.merge_all()

dataset = dataset1.read_train_sets(X_train,train['is_iceberg'],validation_size)

xtest = np.reshape(X_train,[-1,16875])
ytest = train['is_iceberg'].values.reshape([-1,1])

with tf.Session() as sess:
    # variables need to be initialized before we can use them
    sess.run(tf.global_variables_initializer())
    # create log writer object
    writer =  tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    
    for _ in range(10):
        batch_xs, batch_ys = dataset.train.next_batch(100)
        
        batch_xs = np.reshape(batch_xs,[-1,16875])
        batch_ys = batch_ys.values.reshape([-1,1])
        

        val_batch_xs, val_batch_ys = dataset.valid.next_batch(100)
        '''tf.train.batch(
                                    [X_train, train['is_iceberg']],
                                    batch_size=batch_size
                                    #,num_threads=1
                                    ).next_batch(100)'''
        
        _ = sess.run([optimizer], feed_dict={x: batch_xs, y_true: batch_ys})
        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        # write log
        #writer.add_summary(summary, _)
        val_batch_xs = np.reshape(val_batch_xs,[-1,16875])
        val_batch_ys = val_batch_ys.values.reshape([-1,1])
        
        print(sess.run(accuracy, feed_dict={x: val_batch_xs,
                                          y_true: val_batch_ys}))
   
    
            