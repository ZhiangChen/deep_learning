#!/usr/bin/env python3
'''
Zhiang Chen
Aug,2016
'''

"Label the depth images, and store them as the format of 3D volume, which is consistent with the input of convnet"

import os
import numpy as np
from PIL import Image
from scipy import ndimage
from six.moves import cPickle as pickle

pixel_depth = 225.0
image_size = 34
index_image = 0
num_shift = 3  #3x3
stride_shift = 1

lf_x = 16
lf_y = 72
rt_x = lf_x + image_size
rt_y = lf_y + image_size

labels = dict()
label_num = 0

wd = os.getcwd()
print("Current directory is \""+wd+"\"")
print("Start to label the images in this directory? (yes/no)")
cmd = input()
assert cmd == "yes" or cmd == "no"
if cmd == "no":
	print("Input correct directory:")
	wd = input()
	assert os.path.isdir(wd)

input("Before labeling, make sure there are no subdirectories in this folder!\nPress Enter to continue")

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


for name in files:
	debris = name.split('.')
	if debris[-1] != 'bmp':
		continue
	elif name.startswith('cropped'):
		continue
	else:
		print("--------------------------------")
		image_file = os.path.join(wd, name)
		org_img = Image.open(image_file)
		img = org_img.crop((lf_x, lf_y, rt_x, rt_y))
		img.show()
		print("The file name: "+name)
		name = input("Input the name of the object: ")
		face = input("Input the face: ")
		orientation = input("Input the orientation: ")
		
		for i in range(num_shift):
			for j in range(num_shift):
				index = 3*j+i
				# crop image
				img = org_img.crop((lf_x + i*stride_shift, lf_y + j*stride_shift,
					rt_x + i*stride_shift,rt_y + j*stride_shift))
				cropped_name = wd+"/cropped_"+name +'_f'+str(face)+'_r'+str(orientation)+'_'+str(index)+'.bmp'
				img.save(cropped_name)



