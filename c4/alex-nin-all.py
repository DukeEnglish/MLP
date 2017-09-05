#!/disk/scratch/mlp/miniconda2/bin/python
# coding: utf-8

# In[1]:

import os
import datetime
import numpy as np
import tensorflow as tf
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider
import matplotlib.pyplot as plt
import time


# In[2]:

def fully_connected_layer(inputs, input_dim, output_dim, nonlinearity=tf.nn.relu, dropout = True, keep_prob=1.0):
    with tf.device('/cpu:0'):
        weights = tf.Variable(
            tf.truncated_normal(
                [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
            'weights')
        biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = tf.nn.dropout(nonlinearity(tf.matmul(inputs, weights) + biases), keep_prob)
    return outputs

def fully_bn_connected_layer(inputs, input_dim, output_dim, nonlinearity=tf.nn.relu, dropout = True, keep_prob=1.0, train=True):
    with tf.device('/cpu:0'):
        weights = tf.Variable(
            tf.truncated_normal(
                [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
            'weights')
        z_BN = tf.matmul(inputs,weights)
        BN = tf.contrib.layers.batch_norm(z_BN, center=True, scale=True,
                                                 is_training=train)
    outputs = tf.nn.dropout(nonlinearity(BN), keep_prob)
    return outputs

def _variable_with_weight_decay(name, shape, stddev, wd):
    var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32),dtype=tf.float32)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


# In[3]:

def conv_bn_layer1(inputs, nonlinearity=tf.nn.relu, dropout = True, keep_prob=1.0,train=True):
    kernel = _variable_with_weight_decay('weights1',
                                         shape=[3, 3, 3, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    reshape = tf.reshape(inputs, [50,32,32,3])
    conv = tf.nn.conv2d(reshape, kernel, [1, 1, 1, 1], padding='SAME')
    BN2 = tf.contrib.layers.batch_norm(conv, center=True, scale=True,
                                                 is_training=train)
    outputs = tf.nn.dropout(nonlinearity(BN2), keep_prob)
    return outputs

def conv_bn_layer2(inputs, nonlinearity=tf.nn.relu, dropout = True, keep_prob=1.0,train=True):
    kernel = _variable_with_weight_decay('weights2',
                                         shape=[3, 3, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
    BN2 = tf.contrib.layers.batch_norm(conv, center=True, scale=True,
                                                 is_training=train)
    outputs = tf.nn.dropout(nonlinearity(BN2), keep_prob)
    return outputs


# In[4]:

class AugmentedCIFAR10DataProvider(CIFAR10DataProvider):

    def __init__(self, which_set='train', batch_size=50, max_num_batches=-1,
                 shuffle_order=True, rng=None, transformer=None):
        super(AugmentedCIFAR10DataProvider, self).__init__(
            which_set, batch_size, max_num_batches, shuffle_order, rng)
        self.transformer = transformer

    def next(self):
        inputs_batch, targets_batch = super(
            AugmentedCIFAR10DataProvider, self).next()
        transformed_inputs_batch = self.transformer(inputs_batch, self.rng)
        return transformed_inputs_batch, targets_batch


# In[5]:

rng = np.random.RandomState(123456)
def random_fliplr(inputs, rng):
    orig_ims = inputs.reshape((-1, 32, 32, 3))
    new_ims = orig_ims.copy()
    indices = rng.choice(orig_ims.shape[0], orig_ims.shape[0] // 4, False)
    for i, j in enumerate(indices):
        new_ims[j] = np.fliplr(orig_ims[j])
    return new_ims.reshape((-1, 3072))


# In[6]:

random_seed = 1627522
rng = np.random.RandomState(random_seed)

train_data = AugmentedCIFAR10DataProvider('train', batch_size=50, transformer=random_fliplr,rng=rng)
valid_data = CIFAR10DataProvider('valid', batch_size=50, rng=rng)


# In[6]:

inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
keep_prob=tf.placeholder(tf.float32)
train=tf.placeholder(bool)

with tf.variable_scope('conv1') as scope:
    reshape=tf.reshape(inputs,[50,3,32,32])
    reshape=tf.transpose(reshape,[0,2,3,1])
    conv1 = conv_bn_layer1(reshape,keep_prob=1.0)

# pool1
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
# norm1
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

# conv2
with tf.variable_scope('conv2') as scope:
    conv2 = conv_bn_layer2(norm1,keep_prob=1.0)
    #_activation_summary(conv2)

norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
# pool2
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

# local3
with tf.variable_scope('local3') as scope:
    hidden_1=tf.nn.avg_pool(norm2, ksize=[1, 16, 16, 1],
                         strides=[1, 1, 1, 1], padding='VALID', name='pool3')
    # Move everything into depth so we can perform a single matrix multiply.
    
    #hidden_1 = fully_connected_layer(reshape, 4096, 1000, keep_prob=1.0)

# local4
with tf.variable_scope('local4') as scope:
    reshape = tf.reshape(hidden_1, [50,-1])
    outputs=fully_connected_layer(reshape, 64, train_data.num_classes, tf.identity, keep_prob=1.0)
    
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    train_step = tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.9).minimize(error)
    


# In[7]:

init = tf.global_variables_initializer()


train_data.reset()
valid_data.reset()
running_error_one = []
running_accuracy_one = []
valid_error_one = []
valid_accuracy_one = []

with tf.Session() as sess:
    sess.run(init)
    for e in range(200):
        running_error = 0.
        running_accuracy = 0.
        start_time = time.time()
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch, keep_prob:0.8, train: True})
            running_error += batch_error
            running_accuracy += batch_acc
        epoch_time = time.time() - start_time
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        
        running_error_one.append(running_error)
        running_accuracy_one.append(running_accuracy)
        
        print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f} epoch time={3:.2f}'
              .format(e + 1, running_error, running_accuracy, epoch_time))
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch, keep_prob:1.0, train: False})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
            
        valid_error_one.append(valid_error)
        valid_accuracy_one.append(valid_accuracy)
            
        print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                .format(valid_error, valid_accuracy))


# In[ ]:



