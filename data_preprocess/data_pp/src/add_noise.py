#!/usr/bin/env python3
'''
Zhiang Chen
Aug,2016
'''

"Add noises to the cropped depth images"

import os
import numpy as np
from PIL import Image
from scipy import ndimage
from six.moves import cPickle as pickle
import random

pixel_depth = 225.0
image_size = 34
times_noise = 3
num_noise = 10

wd = os.getcwd()
print("Current directory is \""+wd+"\"")
print("Start to add noises to the images in this directory? (yes/no)")
cmd = input()
assert cmd == "yes" or cmd == "no"
if cmd == "no":
	print("Input correct directory:")
	wd = input()
	assert os.path.isdir(wd)

files = os.listdir(wd)
num_images = 0
for name in files:
	if name.startswith('cropped'):
		num_images +=1
print("The number of the cropped images: ", num_images)
num_images *= times_noise
dataset = np.ndarray(shape=(num_images, image_size, image_size),dtype=np.float32)
names = list()
faces = list()
orientations = list()

index = 0
for time in range(times_noise):	
	for name in files:
		if name.startswith('cropped'):
			image_file = os.path.join(wd, name)
			img = Image.open(image_file)

			debris = name.split('_')
			names.append(debris[1])
			faces.append(debris[3][1:])
			orientations.append(debris[4][1:])
			for n in range(num_noise):
				x_pxl = random.randint(0,image_size-1)
				y_pxl = random.randint(0,image_size-1)
				img.putpixel((x_pxl,y_pxl),0)
			image_data = (ndimage.imread(wd+'/'+name).astype(float) - pixel_depth / 2) / pixel_depth
			if image_data.shape != (image_size, image_size):
				raise Exception('Unexpected image shape: %s' % str(image_data.shape))
			dataset[index,:,:] = image_data
			index += 1

print(dataset.shape)		
data_file = wd + '/depth_data'
with open(data_file,'wb') as f:
	save={
		'dataset': dataset,
		'names': names,
		'faces': faces,
		'orientations': orientations
	}
	pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
	f.close()
statinfo = os.stat(data_file)
file_size = float(statinfo.st_size)/1000
print('Compressed data size: %0.1fkB' % file_size)
