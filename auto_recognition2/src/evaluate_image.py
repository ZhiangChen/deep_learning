#!/usr/bin/env python2

# MIT License
#
# Copyright (c) 2016 Zhiang Chen

'''
Receive the cropped image from "cropped_depth_image", and reformat its shape to be compatible with the input shape for the network. 
And then evaluate the input.
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
from auto_recognition.msg import PredictionMSG

name2value = {'v8':0,'ducky':1,'stapler':2,'pball':3,'tball':4,'sponge':5,'bclip':6,'tape':7,'gstick':8,'cup':9,
              'pen':10,'calc':11,'tmeas':12,'bottle':13,'cpin':14,'scissors':15,'stape':16,'gball':17,'orwidg':18,
             'glue':19,'spoon':20,'fork':21,'nerf':22,'eraser':23}
value2name = dict((value,name) for name,value in name2value.items()) 

pixel_depth = 225.0
image_size = 34
num_labels = 24
num_channels = 1
batch_size = 16
patch_size = 5
kernel_size = 2
depth1 = 6 #the depth of 1st convnet
depth2 = 16 #the depth of 2nd convnet
C5_units = 120
F6_units = 84
F7_units = 10

graph = tf.Graph()

with graph.as_default():
	# Input data
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
	# convolution's input is a tensor of shape [batch,in_height,in_width,in_channels]
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_test_dataset = tf.placeholder(tf.float32,shape=(1,image_size,image_size,num_channels))
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
	# C5 actually is a fully-connected layer                        
	C5_weights = tf.Variable(tf.truncated_normal([6 * 6 * depth2, C5_units], stddev=0.1))
	C5_biases = tf.Variable(tf.constant(1.0, shape=[C5_units]))
	F6_weights = tf.Variable(tf.truncated_normal([C5_units,F6_units], stddev=0.1))
	F6_biases = tf.Variable(tf.constant(1.0, shape=[F6_units]))
	# FC and logistic regression replace RBF
	F7_weights = tf.Variable(tf.truncated_normal([F6_units,num_labels], stddev=0.1))
	F7_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
	saver = tf.train.Saver()
	# Model
	def model(data):
		conv = tf.nn.conv2d(data, C1_weights, [1, 1, 1, 1], padding='SAME')
		hidden = tf.nn.relu(conv + C1_biases)
		max_pool = tf.nn.max_pool(hidden,[1,kernel_size,kernel_size,1],[1,2,2,1],'VALID')
		hidden = tf.nn.relu(max_pool)
		conv = tf.nn.conv2d(hidden, C3_weights, [1, 1, 1, 1], padding='VALID')
		hidden = tf.nn.relu(conv + C3_biases)
		max_pool = tf.nn.max_pool(hidden,[1,kernel_size,kernel_size,1],[1,2,2,1],'VALID')
		hidden = tf.nn.relu(max_pool)
		shape = hidden.get_shape().as_list()
		reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
		hidden = tf.nn.relu(tf.matmul(reshape, C5_weights) + C5_biases)
		fc = tf.matmul(hidden,F6_weights)
		hidden = tf.nn.relu(fc + F6_biases)
		fc = tf.matmul(hidden,F7_weights)
		output = fc + F7_biases
		return output
	# Training computation.
	tf_train_dataset = tf.nn.dropout(tf_train_dataset,0.8) # input dropout
	logits = model(tf_train_dataset)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
	# Optimizer.
	optimizer = tf.train.GradientDescentOptimizer(0.0008).minimize(loss)
	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(logits)
	test_prediction = tf.nn.softmax(model(tf_test_dataset))

def accuracy(predictions, labels):
  	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])

class evaluator:
	def __init__(self):
		'''Initialize ros publisher, subscriber'''
		self.pub1  = rospy.Publisher('prediction',PredictionMSG,queue_size=1)
		self.pub2  = rospy.Publisher('pre_image',Image,queue_size=1)
		self.sub1 = rospy.Subscriber('cropped_depth_image',Image,self.callback,queue_size=1)
		self.sub2 = rospy.Subscriber('cropped_depth_image',Image,self.valuate,queue_size=1)
		self.bridge = CvBridge()
		self.got_image = False
		rospy.loginfo("Initialized!")

	def callback(self,data):
		cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="mono8")
		np_image = (cv_image.astype(np.float32) - pixel_depth / 2) / pixel_depth
		self.input_image = np_image.reshape((1,image_size,image_size,num_channels)).astype(np.float32)
		self.got_image = True


	def valuate(self,data):
		config = tf.ConfigProto()
		#config.log_device_placement = True   
		with tf.Session(graph=graph, config = config) as session:
	  		saver.restore(session, "new_model.ckpt")
  			while self.got_image:
		  		self.got_image = False
		  		prediction = tf.nn.softmax(model(self.input_image))
		  		pre_dict = dict(zip(list(range(num_labels)),prediction.eval()[0]))
		        sorted_pre_dict = sorted(pre_dict.items(), key=operator.itemgetter(1))
		        name1 = value2name[sorted_pre_dict[-1][0]]
		        value1 = sorted_pre_dict[-1][1]
		        name2 = value2name[sorted_pre_dict[-2][0]]
		        value2 = sorted_pre_dict[-2][1]
		        pre = PredictionMSG()
		        pre.name1, pre.value1, pre.name2, pre.value2 = name1, value1, name2, value2
		        self.pub1.publish(pre)
		        sys.stdout.write(".")
      			sys.stdout.flush()
      			'''cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="mono8")
      			font = cv2.FONT_HERSHEY_SIMPLEX
        		obj1 = name1+': '+str(value1)
        		obj2 = name2+': '+str(value2)
        		cv2.putText(cv_image,obj1,(0,25),font,0.2,(255,255,255),1)
        		cv2.putText(cv_image,obj2,(0,31),font,0.2,(255,255,255),1)
        		ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="mono8")
        		self.pub2.publish(ros_image)'''
        		#image = self.input_image.reshape((image_size,image_size)).astype(np.float32)
        		#plt.imshow(image,cmap='Greys_r')
        		#self.img_obj.set_data(image)
        		#plt.suptitle(tile, fontsize=12)
        		#plt.draw()
        		#plt.show()
		          	

if __name__ == '__main__':
	rospy.init_node('evaluator',anonymous=True)
	ev = evaluator()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down ROS node evaluate_image")
  		
