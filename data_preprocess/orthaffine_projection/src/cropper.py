#!/usr/bin/env python

import numpy as np
from sensor_msgs.msg import Image
from rospy.numpy_msg import numpy_msg
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
import rospy 
from rospy_tutorials.msg import Floats
import os
import sys
from six.moves import cPickle as pickle

class Cropper():
	def __init__(self):
		self.image_size = rospy.get_param("image_size")
		self.sub = rospy.Subscriber('/box_image/numpy', numpy_msg(Floats), self.callback, queue_size=1)
		self.pub = rospy.Publisher('/cropped_box_image/numpy', numpy_msg(Floats), queue_size=1)
		self.box_size = rospy.get_param("box_size")
		rospy.loginfo("Initialized!")

	def centercrop(self,image_numpy):
		[x,y] = [(l-self.image_size)/2 for l in image_numpy.shape]
		self.center_image = image_numpy[x:x+self.image_size , y:y+self.image_size]

	def randomcrop(self, image_numpy, nm):
		[w,h] = [(l-self.image_size) for l in image_numpy.shape]
		indices = list()
		images = list()
		while len(images)<nm:
			[x,y] = [np.random.randint(0,l) for l in [w,h]]
			if [x,y] not in indices:
				indices.append([x,y])
				images.append(image_numpy[x:x+self.image_size , y:y+self.image_size])
		return images

	def set_boxes(self,bias):
		'bias representing the bias of the center of box w.r.t the center of image is a 2D list with shape (-1,2)'
		center = np.asarray([self.image_size/2, self.image_size/2])
		bias = np.asarray(bias)
		delta = np.asarray([(self.box_size - self.image_size)/2, (self.box_size - self.image_size)/2])
		self.xy = delta + bias

	def arraycrop(self,image_numpy,save_crop=False):
		images = [image_numpy[x:x+self.image_size , y:y+self.image_size] for x,y in self.xy]
		return images

	def crop_pickle(self,pickle_name,func=1,bias=[[0,0]],nm=5):
		'''
		func=1: arraycrop
		func=2: randomcrop
		'''
		with open(pickle_name,'rb') as f:
			save = pickle.load(f):
			data = save['image']
			del save
		images = dict()
		if func == 1:
			self.set_boxes(bias)
			for name, value in data.iteritems():
				cropped_images = self.arraycrop(value)
				for key, cropped_image in enumerate(cropped_images):
					images.setdefault(str(key)+'-'+name,cropped_image)

		elif func == 2:
			for name, value in data.iteritems():
				cropped_images = self.randomcrop(value,nm)
				for key, cropped_image in enumerate(cropped_images):
					images.setdefault(str(key)+'-'+name,cropped_image)

		with open('new_'+pickle_name,'wb') as f:
			save={
			'image': images
			}
			pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
			f.close()				





	def callback(self, image_array):
		image_numpy = image_array.data.reshape(self.box_size,self.box_size)
		self.centercrop(image_numpy)
		self.pub.publish(self.center_image.reshape(-1))
		sys.stdout.write(".")
		sys.stdout.flush()

def SaveRandomCropped(image_size,nm):
	cropper = Cropper(image_size)
	with open('depth_data_numpy', 'rb') as f:
		save = pickle.load(f)
		images = save['image']
		del save
	cropped_images = dict()
	for name,image in images.items():
		rim = np.asarray(cropper.randomcrop(image,nm))
		cropped_images.setdefault(name,rim)
	with open('cropped_depth_data_numpy','wb') as f:
		save={
		'image': cropped_images
		}
		pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
		f.close()
	statinfo = os.stat('cropped_depth_data_numpy')
	file_size = float(statinfo.st_size)/1000
	print('Compressed data size: %0.1fkB' % file_size)

if __name__=='__main__':
	rospy.init_node('Cropper',anonymous=True)
	cropper = Cropper(80)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down ROS node Cropper")
