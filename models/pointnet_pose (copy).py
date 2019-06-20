
# THIS IS A DEMO CODE FOR THE FOLLOWING PAPER 
# Synthetic Views Generation for Absolute Pose Regression and Virtual Reality, BMVC 2018 
# DEVELOPED BY PULAK PURKAIT, EMAIL: pulak.purkait@crl.toshiba.co.uk 


import tensorflow as tf
import numpy as np
import math
import sys
import os
# import helper

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
#from transform_nets import input_transform_net, feature_transform_net
import tf_util

def placeholder_inputs(batch_size, BLOCK_SIZE1, BLOCK_SIZE2):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, BLOCK_SIZE1, BLOCK_SIZE2, 5+128)) 
   # 2 is for the pixel locations and 336 for the SIFT feature vectors 
   # pointclouds_3d = tf.placeholder(tf.float32, shape=(batch_size, BLOCK_SIZE1, BLOCK_SIZE2, 3)) 
    labels_pose = tf.placeholder(tf.float32, shape=(batch_size, 3+4))
    return pointclouds_pl, labels_pose

def get_model(point_cloud, is_training, bn_decay=None):
    """ Pose Regression using PointNet, input is BxNx3, output Bx40 """
    #print(point_cloud.shape())
    batch_size = point_cloud.get_shape()[0].value
    BLOCK_SIZE1 = point_cloud.get_shape()[1].value
    BLOCK_SIZE2 = point_cloud.get_shape()[2].value
    
    #print batch_size, num_point, dim_point 
    pixel_points = point_cloud[:, :, :, :2]

    input_image = point_cloud
    net1 = tf_util.conv2d(input_image, 128, [1, 1], # 3 is replaced by two 
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv11', bn_decay=bn_decay)
    #### Net1  
    net1 = tf_util.conv2d(net1, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv12', bn_decay=bn_decay)
    

    net1 = tf_util.max_pool2d(net1, [4,4], stride=[4,4], 
                            padding='VALID', scope='maxpool12')   

    net1 = tf_util.conv2d(net1, 196, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv15', bn_decay=bn_decay) 
    net1 = tf_util.conv2d(net1, 96, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv16', bn_decay=bn_decay)


    #### Net2 


    net2 = tf_util.conv2d(input_image, 128, [1, 1], # 3 is replaced by two 
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv21', bn_decay=bn_decay)
    net2 = tf_util.conv2d(net2, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv22', bn_decay=bn_decay)
    

    net2 = tf_util.max_pool2d(net2, [4,4], stride=[4,4], 
                            padding='VALID', scope='maxpool22')   
    net2 = tf_util.conv2d(net2, 336, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv25', bn_decay=bn_decay) 
    net2 = tf_util.conv2d(net2, 336, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv26', bn_decay=bn_decay)


    #### Net3  


    net3 = tf_util.conv2d(input_image, 128, [1, 1], # 3 is replaced by two 
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv31', bn_decay=bn_decay)
    net3 = tf_util.conv2d(net3, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv32', bn_decay=bn_decay)
    

    net3 = tf_util.max_pool2d(net3, [4,4], stride=[4,4], 
                            padding='VALID', scope='maxpool32')   
    net3 = tf_util.conv2d(net3, 778, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv35', bn_decay=bn_decay) 
    net3 = tf_util.conv2d(net3, 1536, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv36', bn_decay=bn_decay)




    net1 = tf_util.max_pool2d(net1, [2,2], stride=[2,2], 
                            padding='VALID', scope='maxpool3') 

    #### Concatenation of Net1, Net2, Net3 

    net2 = tf_util.max_pool2d(net2, [4,4], stride=[4,4], 
                            padding='VALID', scope='maxpool4') 
    #print net2.shape  

    net3 = tf_util.max_pool2d(net3, [8,8], stride=[1,1], 
                          padding='VALID', scope='maxpool5') 

    net1 = tf.reshape(net1, [batch_size, -1])
    net2 = tf.reshape(net2, [batch_size, -1])
    net3 = tf.reshape(net3, [batch_size, -1])
    net = tf.concat([net1, net2, net3], 1)
    

    #### Fully Connected Layers - DropOut  --- Bigger Version 

    #### Try with smaller Network, i.e, 1024 parameters 

    net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                          scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='dp1')

    net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                          scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='dp2')

    nett = tf_util.fully_connected(net, 120, bn=True, is_training=is_training,
                          scope='fc5', bn_decay=bn_decay)
    netr = tf_util.fully_connected(net, 120, bn=True, is_training=is_training,
                          scope='fc6', bn_decay=bn_decay)

    nett = tf_util.fully_connected(nett, 3, activation_fn=None, scope='fc7')
    netr = tf_util.fully_connected(netr, 4, activation_fn=None, scope='fc8')

    net = tf.concat([nett, netr], 1)

    return net, pixel_points # Note that pixel co-ordinates are not used directly during the training 


def get_loss(y_pred, y_true, pixel_points, sigx, sigy, reg_weight=0.001):

    predicted_q = y_pred[:, 3:]
    predicted_q = tf.nn.l2_normalize(predicted_q, dim=1)

    error = sigx+sigy+tf.exp(tf.multiply(-1.0, sigx))*tf.reduce_mean((tf.reduce_mean(tf.abs(tf.subtract(y_true[:, :3], y_pred[:, :3])), axis=1))) + tf.exp(tf.multiply(-1.0, sigy))*tf.reduce_mean((tf.reduce_mean(tf.abs(tf.subtract(y_true[:, 3:], predicted_q)), axis=1))) 

    tf.summary.scalar('mat loss', error)

    return error 

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((10, 32, 32, 133))
        outputs = get_model(inputs, tf.constant(True))
	print outputs


