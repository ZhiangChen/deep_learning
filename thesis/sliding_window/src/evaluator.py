#!/usr/bin/env python2

# MIT License
#
# Copyright (c) 2017 Zhiang Chen
'''
Receive the cropped image from "box_image/numpy", and publish the class prediction and angle prediction onto "prediction"
'''
from __future__ import print_function
import rospy
import roslib
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import operator
import time
import os
import sys
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from depthnet.msg import PredictionMSG
import math
from six.moves import cPickle as pickle

with open('small_data', 'rb') as f:
    save = pickle.load(f)
    small_data = save['small_data']
    del save

name2value = {'empty':0,'duck':1,'cup':2,'sponge':3,'tball':4,'pball':5,'gball':6,'gstick':7,'nerf':8,'calc':9,'stapler':10}
value2name = dict((value,name) for name,value in name2value.items()) 

name2string = {'v8':'v8 can','duck':'ducky','stapler':'stapler','pball':'ping pang ball','tball':'tennis ball','sponge':'sponge',
            'bclip':'binder clip','tape':'big tape','gstick':'glue stick','cup':'cup','pen':'pen','calc':'calculator',
            'blade':'razor','bottle':'bottle','cpin':'clothespin','scissors':'scissors','stape':'small tape','gball':'golf ball',
            'orwidg':'orange thing','glue':'glue','spoon':'spoon','fork':'fork','nerf':'nerf gun','eraser':'eraser',
            'empty':'empty plate'}

angles_list = np.asarray([i*18 for i in range(10)]).astype(np.float32)

num_labels = 11
image_size = 50
'''ConvNet'''
k1_size = 6
k1_stride = 1
k1_depth = 1
k1_nm = 16
n1 = image_size*image_size*1

k2_size = 3
k2_stride = 2
k2_depth = 16
k2_nm = 16
m1_size = image_size-k1_size+k1_stride
n2 = m1_size*m1_size*k1_nm

k3_size = 6
k3_stride = 1
k3_depth = 16
k3_nm = 32
m2_size = (m1_size-k2_size)/k2_stride+1
n3 = m2_size*m2_size*k2_nm

k4_size = 3
k4_stride = 2
k4_depth = 32
k4_nm = 32
m3_size = (m2_size-k3_size)/k3_stride+1
n4 = m3_size*m3_size*k3_nm

k5_size = 3
k5_stride = 1
k5_depth = 32
k5_nm = 64
m4_size = (m3_size-k4_size)/k4_stride+1
n5 = m4_size*m4_size*k4_nm

k6_size = 2
k6_stride = 2
k6_depth = 64
k6_nm = 64
m5_size = (m4_size-k5_size)/k5_stride+1
n6 = m5_size*m5_size*k5_nm

'''Class FC'''
f7_class_size = 120
m6_class_size = (m5_size-k6_size)/k6_stride+1
n7_class = m6_class_size*m6_class_size*k6_nm

f8_class_size = 60
n8_class = f7_class_size

classes_size = 11
n9_class = f8_class_size

'''Angle FC'''
f7_angle_size = 120
m6_angle_size = (m5_size-k6_size)/k6_stride+1
n7_angle = m6_angle_size*m6_angle_size*k6_nm

f8_angle_size = 60
n8_angle = f7_angle_size

angles_size = 10
n9_angle = f8_angle_size

'''Dropout'''
keep_prob1 = 0.8
keep_prob2 = 0.5

'''Mini-batch'''
batch_size = 33
angles_list = np.asarray([i*18 for i in range(10)]).astype(np.float32)

def leaky_relu(x, leak=0.1):
    return tf.maximum(x, x * leak)

graph = tf.Graph()

with graph.as_default():
    '''Input data'''
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, k1_depth))
    # k1_depth = input_channels
    # convolution's input is a tensor of shape [batch,in_height,in_width,in_channels]
    tf_train_classes = tf.placeholder(tf.float32, shape=(batch_size, 11))
    tf_train_angles = tf.placeholder(tf.float32, shape=(batch_size, 10))
    
    '''Xavier initialization'''
    k1_stddev = math.sqrt(1.0/n1)
    k1_weights = tf.Variable(tf.truncated_normal([k1_size, k1_size, k1_depth, k1_nm], stddev = k1_stddev))
    k1_biases = tf.Variable(tf.zeros([k1_nm]))
    
    k2_stddev = math.sqrt(2.0/n2)
    k2_weights = tf.Variable(tf.truncated_normal([k2_size, k2_size, k2_depth, k2_nm], stddev = k2_stddev))
    k2_biases = tf.Variable(tf.zeros([k2_nm]))
    
    k3_stddev = math.sqrt(2.0/n3)
    k3_weights = tf.Variable(tf.truncated_normal([k3_size, k3_size, k3_depth, k3_nm], stddev = k3_stddev))
    k3_biases = tf.Variable(tf.zeros([k3_nm]))
    
    k4_stddev = math.sqrt(2.0/n4)
    k4_weights = tf.Variable(tf.truncated_normal([k4_size, k4_size, k4_depth, k4_nm], stddev = k4_stddev))
    k4_biases = tf.Variable(tf.zeros([k4_nm]))
    
    k5_stddev = math.sqrt(2.0/n5)
    k5_weights = tf.Variable(tf.truncated_normal([k5_size, k5_size, k5_depth, k5_nm], stddev = k5_stddev))
    k5_biases = tf.Variable(tf.zeros([k5_nm]))
    
    k6_stddev = math.sqrt(2.0/n6)
    k6_weights = tf.Variable(tf.truncated_normal([k6_size, k6_size, k6_depth, k6_nm], stddev = k6_stddev))
    k6_biases = tf.Variable(tf.zeros([k6_nm]))
    
    ## class FC
    f7_class_stddev = math.sqrt(2.0/n7_class)
    f7_class_weights = tf.Variable(tf.truncated_normal([n7_class, f7_class_size], stddev = f7_class_stddev))
    f7_class_biases = tf.Variable(tf.zeros([f7_class_size]))
    
    f8_class_stddev = math.sqrt(2.0/n8_class)
    f8_class_weights = tf.Variable(tf.truncated_normal([n8_class, f8_class_size], stddev = f8_class_stddev))
    f8_class_biases = tf.Variable(tf.zeros([f8_class_size]))
    
    f9_class_stddev = math.sqrt(2.0/n9_class)
    f9_class_weights = tf.Variable(tf.truncated_normal([n9_class, classes_size], stddev = f9_class_stddev))
    f9_class_biases = tf.Variable(tf.zeros([classes_size]))
    
    ## angle FC
    f7_angle_stddev = math.sqrt(2.0/n7_angle)
    f7_angle_weights = tf.Variable(tf.truncated_normal([n7_angle, f7_angle_size], stddev = f7_angle_stddev))
    f7_angle_biases = tf.Variable(tf.zeros([f7_angle_size]))
    
    f8_angle_stddev = math.sqrt(2.0/n8_angle)
    f8_angle_weights = tf.Variable(tf.truncated_normal([n8_angle, f8_angle_size], stddev = f8_angle_stddev))
    f8_angle_biases = tf.Variable(tf.zeros([f8_angle_size]))
    
    f9_angle_stddev = math.sqrt(2.0/n9_angle)
    f9_angle_weights = tf.Variable(tf.truncated_normal([n9_angle, angles_size], stddev = f9_angle_stddev))
    f9_angle_biases = tf.Variable(tf.zeros([angles_size]))
    
    #print n1,n2,n3,n4,n5,n6,n7,n8,n9
    #print k1_stddev,k2_stddev,k3_stddev,k4_stddev,k5_stddev,k6_stddev,f7_stddev,f8_stddev,f9_stddev
    
    '''Batch normalization initialization'''
    beta1 = tf.Variable(tf.zeros([k1_nm]))
    gamma1 = tf.Variable(tf.ones([k1_nm]))
    
    beta2 = tf.Variable(tf.zeros([k2_nm]))
    gamma2 = tf.Variable(tf.ones([k2_nm]))
    
    beta3 = tf.Variable(tf.zeros([k3_nm]))
    gamma3 = tf.Variable(tf.ones([k3_nm]))
    
    beta4 = tf.Variable(tf.zeros([k4_nm]))
    gamma4 = tf.Variable(tf.ones([k4_nm]))

    beta5 = tf.Variable(tf.zeros([k5_nm]))
    gamma5 = tf.Variable(tf.ones([k5_nm]))
    
    beta6 = tf.Variable(tf.zeros([k6_nm]))
    gamma6 = tf.Variable(tf.ones([k6_nm]))

    saver = tf.train.Saver()
    # Model
    def test_model(data):
        conv = tf.nn.conv2d(data, k1_weights, [1, 1, 1, 1], padding='VALID')
        mean, variance = tf.nn.moments(conv, [0, 1, 2])
        y = tf.nn.batch_normalization(conv,mean,variance,beta1,gamma1,1e-5)
        hidden = leaky_relu(y)
        
        conv = tf.nn.conv2d(hidden, k2_weights, [1, 2, 2, 1], padding='VALID')
        mean, variance = tf.nn.moments(conv, [0, 1, 2])
        y = tf.nn.batch_normalization(conv,mean,variance,beta2,gamma2,1e-5)
        hidden = leaky_relu(y)
     
        conv = tf.nn.conv2d(hidden, k3_weights, [1, 1, 1, 1], padding='VALID')
        mean, variance = tf.nn.moments(conv, [0, 1, 2])
        y = tf.nn.batch_normalization(conv,mean,variance,beta3,gamma3,1e-5)
        hidden = leaky_relu(y)
         
        conv = tf.nn.conv2d(hidden, k4_weights, [1, 2, 2, 1], padding='VALID')
        mean, variance = tf.nn.moments(conv, [0, 1, 2])
        y = tf.nn.batch_normalization(conv,mean,variance,beta4,gamma4,1e-5)
        hidden = leaky_relu(y)
        
        conv = tf.nn.conv2d(hidden, k5_weights, [1, 1, 1, 1], padding='VALID')
        mean, variance = tf.nn.moments(conv, [0, 1, 2])
        y = tf.nn.batch_normalization(conv,mean,variance,beta5,gamma5,1e-5)
        hidden = leaky_relu(y)
      
        conv = tf.nn.conv2d(hidden, k6_weights, [1, 2, 2, 1], padding='VALID')
        mean, variance = tf.nn.moments(conv, [0, 1, 2])
        y = tf.nn.batch_normalization(conv,mean,variance,beta6,gamma6,1e-5)
        hidden = leaky_relu(y)
  
        shape = hidden.get_shape().as_list()
        hidden_input = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        
        ## class FC
        class_hidden = leaky_relu(tf.matmul(hidden_input, f7_class_weights) + f7_class_biases)
        class_fc = tf.matmul(class_hidden,f8_class_weights)
        class_hidden = leaky_relu(class_fc + f8_class_biases)
        fc_classes = tf.matmul(class_hidden,f9_class_weights)
        output_classes = fc_classes + f9_class_biases
        
        ## angle FC
        angle_hidden = leaky_relu(tf.matmul(hidden_input, f7_angle_weights) + f7_angle_biases)
        angle_fc = tf.matmul(angle_hidden,f8_angle_weights)
        angle_hidden = leaky_relu(angle_fc + f8_angle_biases)
        fc_angles = tf.matmul(angle_hidden,f9_angle_weights)
        output_angles = fc_angles + f9_angle_biases  
        
        return output_classes, output_angles


config = tf.ConfigProto()
#config.log_device_placement = True 
session = tf.Session(graph=graph, config = config)
saver.restore(session, "./model.ckpt")

def accuracy_classes(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])
'''
from evaluator import *
ev = evaluator()
c,s,a = ev.evaluate(small_data)
'''
class evaluator:
    def __init__(self):
        #Initialize ros publisher, subscriber
        self.pub1  = rospy.Publisher('prediction',PredictionMSG,queue_size=1)
        self.sub1 = rospy.Subscriber('box_image/numpy',numpy_msg(Floats),self.callback,queue_size=1)
        self.pub2  = rospy.Publisher('p_box_image/image',Image, queue_size=1)
        self.pub3 = rospy.Publisher('predicted_class', String, queue_size=1)
        self.bridge = CvBridge()
        self.pt1x = -25.0
        self.pt1y = 0.0
        self.pt2x = 25.0
        self.pt2y = 0.0
        rospy.loginfo("Evaluator Initialized!")

    def evaluate(self,images):
        'images has numpy.ndarray type; images.shape=[-1,image_size,image_size,1]; the elements have np.float32 type' 
        with session.as_default():
            out_class, out_angle = test_model(images)
            pre_class, pre_angle = tf.nn.softmax(out_class).eval(), tf.nn.softmax(out_angle).eval()
            angles = np.sum(np.multiply(pre_angle, angles_list),axis=1)/np.sum(pre_angle,axis=1)
            classes = np.argmax(pre_class, axis=1)+0.1
            #print(classes)
            #classes = [value2name[value] for value in classes]
            #print(classes)
            #classes = [name2string[name] for name in classes]
            scores = np.amax(pre_class,axis=1)
            return classes.reshape(-1,1), scores.reshape(-1,1), angles.reshape(-1,1)



    def callback(self,data):
        with session.as_default():
            assert tf.get_default_session() is session
            input_image = np.flipud(data.data.reshape(image_size,image_size).astype(np.float32)).reshape(-1,image_size,image_size,1)
            images = np.append(input_image,small_data,axis=0)
            out_class, out_angle = test_model(images)
            pre_class = tf.nn.softmax(out_class)
            pre_angle = tf.nn.softmax(out_angle).eval()[0]
            angle = np.sum(np.multiply(pre_angle, angles_list))/np.sum(pre_angle)
            pre_dict = dict(zip(list(range(num_labels)),pre_class.eval()[0]))
            sorted_pre_dict = sorted(pre_dict.items(), key=operator.itemgetter(1))
            name1 = value2name[sorted_pre_dict[-1][0]]
            name1 = name2string[name1]
            self.pub3.publish(name1)
            value1 = str(sorted_pre_dict[-1][1])
            name2 = value2name[sorted_pre_dict[-2][0]]
            name2 = name2string[name2]
            value2 = str(sorted_pre_dict[-2][1])
            pre = PredictionMSG()
            pre.name1, pre.value1, pre.name2, pre.value2, pre.angle = name1, float(value1), name2, float(value2), angle
            self.pub1.publish(pre)
            image = ((input_image.reshape(image_size,image_size) + 0.65)*255).astype(np.uint8)
            pt1x = int(self.pt1x * math.cos(math.radians(angle)) + self.pt1y * -math.sin(math.radians(angle))) + 25
            pt1y = int(self.pt1x * math.sin(math.radians(angle)) + self.pt1y * math.cos(math.radians(angle))) + 25
            pt2x = int(self.pt2x * math.cos(math.radians(angle)) + self.pt2y * -math.sin(math.radians(angle))) + 25
            pt2y = int(self.pt2x * math.sin(math.radians(angle)) + self.pt2y * math.cos(math.radians(angle))) + 25
            cv2.line(image,(pt1x,pt1y),(pt2x,pt2y),150,1)
            ros_image = self.bridge.cv2_to_imgmsg(image, encoding="mono8")
            self.pub2.publish(ros_image)
            sys.stdout.write(".")
            sys.stdout.flush()

if __name__ == '__main__':
    rospy.init_node('multitask',anonymous=True)
    ev = evaluator()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS node evaluate_image")
    session.close()
    print("Shutting down ROS node evaluate_image")
