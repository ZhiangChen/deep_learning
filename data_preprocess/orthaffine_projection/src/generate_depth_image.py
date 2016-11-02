#!/usr/bin/env python

import pcl
import cv2
import numpy as np
import matplotlib.pyplot as plt
from orthaffine import OrthAffine as OA
import os
from math import *
from six.moves import cPickle as pickle

files = os.listdir('.')
pcd_files = list()
for f in files:
	if f.startswith('box'):
		if (f.split('_')[0]=='box') & (f.split('.')[1]=='pcd'):
			pcd_files.append(f)

nm = len(pcd_files)
print('There are %d files to be processed' % nm)
theta = 30.0/180*pi 
oa = OA(theta)
images = dict()
for f in pcd_files:
	oa.readpcd(f)
	oa.affine()
	oa.interpolate(theta)
	oa.project()
	name = '-'.join(f.split('.')[0].split('_')[1:])+'.png'
	images.setdefault(name,oa.image_numpy)
	#oa.saveimage(name)
	#oa.savepcd('box.pcd')

wd = os.getcwd()
data_file = wd + '/depth_data_numpy'
with open(data_file,'wb') as f:
	save={
		'image': images
	}
	pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
	f.close()
statinfo = os.stat(data_file)
file_size = float(statinfo.st_size)/1000
print('Compressed data size: %0.1fkB' % file_size)
