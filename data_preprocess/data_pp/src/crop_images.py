#!/usr/bin/env python2
'''
Zhiang Chen
Aug,2016
'''

"Label the depth images, and store them as the format of 3D volume, which is consistent with the input of convnet"

import os
import numpy as np
from PIL import Image
from scipy import ndimage
import rospy

pixel_depth = 225.0
index_image = 0
num_shift = 3  #3x3
stride_shift = 1

lf_x = 16
lf_y = 72
lf_x = rospy.get_param("tplf_x")
lf_y = rospy.get_param("tplf_y")
rt_x = rospy.get_param("btrt_x")
rt_y = rospy.get_param("btrt_y")
image_size = rospy.get_param("image_size")
filter = Image.ANTIALIAS

labels = dict()
label_num = 0

wd = os.getcwd()
print("Current directory is \""+wd+"\"")
print("Start to label the images in this directory? (yes/no)")
cmd = raw_input()
assert cmd == "yes" or cmd == "no"
if cmd == "no":
	print("Input correct directory:")
	wd = input()
	assert os.path.isdir(wd)

raw_input("Before labeling, make sure there are no subdirectories in this folder!\nPress Enter to continue")

files = os.listdir(wd)

num_images = 0
for name in files:
	debris = name.split('.')
	if debris[-1] != 'bmp':
		continue
	elif name.startswith('cropped'):
		continue
	else:
		num_images += 1
print("There are %d images to label" %num_images)


count = 0
for name in files:
	debris = name.split('.')
	if debris[-1] != 'bmp':
		continue
	elif name.startswith('cropped'):
		continue
	else:
		image_file = os.path.join(wd, name)
		org_img = Image.open(image_file)
		img = org_img.crop((lf_x, lf_y, rt_x, rt_y))
#		img.show()
		org_info = debris[0].split('-')
		info = org_info[0].split('_')
#		print("The file name: "+name)
#		name = input("Input the name of the object: ")
#		face = input("Input the face: ")
#		orientation = input("Input the orientation: ")
		position = info[1]
		name = info[2]
		phase = 0
		orientation = info[3] 
		for i in range(num_shift):
			for j in range(num_shift):
				index = 3*j+i
				# crop image
				img = org_img.crop((lf_x + i*stride_shift, lf_y + j*stride_shift,
					rt_x + i*stride_shift,rt_y + j*stride_shift)).resize((image_size,image_size),filter)
				cropped_name = wd+"/cropped_"+name+'_p'+str(position)+'_f'+str(phase)+'_r'+str(orientation)+'_'+str(index)+'.bmp'
				img.save(cropped_name)

		count += 1
print("Done!")



