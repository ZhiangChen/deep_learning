#!/usr/bin/env python2

# MIT License
#
# Copyright (c) 2016 Zhiang Chen
'''
Receive the cropped image from "cropped_box_image/numpy", and publish the class prediction and angle prediction onto "prediction"
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

name2value = {'v8':0,'duck':1,'stapler':2,'pball':3,'tball':4,'sponge':5,'bclip':6,'tape':7,'gstick':8,'cup':9,
            'pen':10,'calc':11,'blade':12,'bottle':13,'cpin':14,'scissors':15,'stape':16,'gball':17,'orwidg':18,
            'glue':19,'spoon':20,'fork':21,'nerf':22,'eraser':23,'empty':24}
name2string = {'v8':'v8 can','duck':'ducky','stapler':'stapler','pball':'ping pang ball','tball':'tennis ball','sponge':'sponge',
            'bclip':'binder clip','tape':'big tape','gstick':'glue stick','cup':'cup','pen':'pen','calc':'calculator',
            'blade':'razor','bottle':'bottle','cpin':'clothespin','scissors':'scissors','stape':'small tape','gball':'golf ball',
            'orwidg':'orange thing','glue':'glue','spoon':'spoon','fork':'fork','nerf':'nerf gun','eraser':'eraser',
            'empty':'empty plate'}
value2name = dict((value,name) for name,value in name2value.items()) 

image_size = 80
num_labels = 25
angle_bias = -15
nm_classes = 25
nm_angles = 10
num_channels = 1 

batch_size = 30
patch_size = 5
kernel_size = 2
depth1 = 6
depth2 = 16 
depth3 = 10 
F7_classes = 120
F8_classes = 84
F9_classes = nm_classes
F7_angles = 120
F8_angles = 84
F9_angles = nm_angles

keep_prob1 = 0.5
keep_prob2_classes = 0.8
keep_prob2_angles = 0.5
angles_list = np.asarray([i*18 + angle_bias for i in range(10)]).astype(np.float32)

graph = tf.Graph()

with graph.as_default():
    # Variables(weights and biases)
    C1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth1], stddev=0.1))
    # convolution's weights are called filter in tensorflow
    # it is a tensor of shape [kernel_hight,kernel_width,in_channels,out_channels]
    C1_biases = tf.Variable(tf.zeros([depth1]))

    # S1_weights # Sub-sampling doesn't need weights and biases
    # S1_biases

    C3_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth1, depth2], stddev=0.1))
    C3_biases = tf.Variable(tf.constant(1.0, shape=[depth2]))

    # S4_weights
    # S4_biases

    C5_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth2, depth3], stddev=0.1))
    C5_biases = tf.Variable(tf.constant(1.0, shape=[depth3]))

    # S6_weights
    # S6_biases

    F7_classes_weights = tf.Variable(tf.truncated_normal([6 * 6 * depth3, F7_classes], stddev=0.1))
    F7_classes_biases = tf.Variable(tf.constant(1.0, shape=[F7_classes]))
    F7_angles_weights = tf.Variable(tf.truncated_normal([6 * 6 * depth3, F7_angles], stddev=0.1))
    F7_angles_biases = tf.Variable(tf.constant(1.0, shape=[F7_angles]))

    F8_classes_weights = tf.Variable(tf.truncated_normal([F7_classes,F8_classes], stddev=0.1))
    F8_classes_biases = tf.Variable(tf.constant(1.0, shape=[F8_classes]))
    F8_angles_weights = tf.Variable(tf.truncated_normal([F7_angles,F8_angles], stddev=0.1))
    F8_angles_biases = tf.Variable(tf.constant(1.0, shape=[F8_angles]))

    F9_classes_weights = tf.Variable(tf.truncated_normal([F8_classes,F9_classes], stddev=0.1))
    F9_classes_biases = tf.Variable(tf.constant(1.0, shape=[F9_classes]))
    F9_angles_weights = tf.Variable(tf.truncated_normal([F8_angles,F9_angles], stddev=0.1))
    F9_angles_biases = tf.Variable(tf.constant(1.0, shape=[F9_angles]))

    saver = tf.train.Saver()
    # Model
    def test_model(data):
        conv = tf.nn.conv2d(data, C1_weights, [1, 1, 1, 1], padding='VALID')
        hidden = tf.nn.relu(conv + C1_biases)

        max_pool = tf.nn.max_pool(hidden,[1,kernel_size,kernel_size,1],[1,2,2,1],'VALID')
        hidden = tf.nn.relu(max_pool)

        conv = tf.nn.conv2d(hidden, C3_weights, [1, 1, 1, 1], padding='VALID')
        hidden = tf.nn.relu(conv + C3_biases)

        max_pool = tf.nn.max_pool(hidden,[1,kernel_size,kernel_size,1],[1,2,2,1],'VALID')
        hidden = tf.nn.relu(max_pool)

        conv = tf.nn.conv2d(hidden,C5_weights, [1,1,1,1], padding = 'VALID')
        hidden = tf.nn.relu(conv + C5_biases)

        max_pool = tf.nn.max_pool(hidden,[1,kernel_size,kernel_size,1],[1,2,2,1],'VALID')
        hidden = tf.nn.relu(max_pool)

        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden_classes = tf.nn.relu(tf.matmul(reshape, F7_classes_weights) + F7_classes_biases)
        hidden_angles = tf.nn.relu(tf.matmul(reshape, F7_angles_weights) + F7_angles_biases)

        fc_classes = tf.matmul(hidden_classes,F8_classes_weights)
        fc_angles = tf.matmul(hidden_angles,F8_angles_weights)
        hidden_classes = tf.nn.relu(fc_classes + F8_classes_biases)
        hidden_angles = tf.nn.relu(fc_angles + F8_angles_biases)

        fc_classes = tf.matmul(hidden_classes,F9_classes_weights)
        fc_angles = tf.matmul(hidden_angles,F9_angles_weights)
        output_classes = fc_classes + F9_classes_biases
        output_angles = fc_angles + F9_angles_biases

        return output_classes, output_angles


config = tf.ConfigProto()
#config.log_device_placement = True 
session = tf.Session(graph=graph, config = config)
saver.restore(session, "model2.ckpt")

def accuracy_classes(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])


class evaluator:
    def __init__(self):
        #Initialize ros publisher, subscriber
        self.pub1  = rospy.Publisher('prediction',PredictionMSG,queue_size=1)
        self.sub1 = rospy.Subscriber('cropped_box_image/numpy',numpy_msg(Floats),self.callback,queue_size=1)
        self.pub2  = rospy.Publisher('cropped_box_image/image',Image, queue_size=1)
        self.bridge = CvBridge()
        self.pt1x = -40.0
        self.pt1y = 0.0
        self.pt2x = 40.0
        self.pt2y = 0.0
        rospy.loginfo("Initialized!")
    def callback(self,data):
        with session.as_default():
            assert tf.get_default_session() is session
            input_image = np.flipud(data.data.reshape(image_size,image_size).astype(np.float32)).reshape(-1,image_size,image_size,1)
            out_class, out_angle = test_model(input_image)
            pre_class = tf.nn.softmax(out_class)
            pre_angle = tf.nn.softmax(out_angle).eval()
            angle = np.sum(np.multiply(pre_angle, angles_list))/np.sum(pre_angle)
            pre_dict = dict(zip(list(range(num_labels)),pre_class.eval()[0]))
            sorted_pre_dict = sorted(pre_dict.items(), key=operator.itemgetter(1))
            name1 = value2name[sorted_pre_dict[-1][0]]
            name1 = name2string[name1]
            value1 = str(sorted_pre_dict[-1][1])
            name2 = value2name[sorted_pre_dict[-2][0]]
            name2 = name2string[name2]
            value2 = str(sorted_pre_dict[-2][1])
            pre = PredictionMSG()
            pre.name1, pre.value1, pre.name2, pre.value2, pre.angle = name1, float(value1), name2, float(value2), angle
            self.pub1.publish(pre)
            image = ((input_image.reshape(image_size,image_size) + 0.65)*255).astype(np.uint8)
            pt1x = int(self.pt1x * math.cos(math.radians(angle)) + self.pt1y * -math.sin(math.radians(angle))) + 40
            pt1y = int(self.pt1x * math.sin(math.radians(angle)) + self.pt1y * math.cos(math.radians(angle))) + 40
            pt2x = int(self.pt2x * math.cos(math.radians(angle)) + self.pt2y * -math.sin(math.radians(angle))) + 40
            pt2y = int(self.pt2x * math.sin(math.radians(angle)) + self.pt2y * math.cos(math.radians(angle))) + 40
            cv2.line(image,(pt1x,pt1y),(pt2x,pt2y),255,2)
            ros_image = self.bridge.cv2_to_imgmsg(image, encoding="mono8")
            self.pub2.publish(ros_image)
            sys.stdout.write(".")
            sys.stdout.flush()

if __name__ == '__main__':
    rospy.init_node('depthnet',anonymous=True)
    ev = evaluator()
try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting down ROS node evaluate_image")

session.close()
print("Shutting down ROS node evaluate_image")
