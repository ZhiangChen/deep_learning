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

images = dict()
for theta_ in [25,30,35]:
	theta = float(theta_)/180*pi 
	oa = OA(theta)
	prefix = str(theta_)+'-'
	for f in pcd_files:
		oa.readpcd(f)
		oa.affine()
		oa.interpolate(theta)
		oa.project()
		name = prefix+'-'.join(f.split('.')[0].split('_')[1:])+'.png'
		images.setdefault(name,oa.image_numpy)
		#oa.savepcd('box.pcd')
	#oa.saveimage(name)
	print(theta_)

wd = os.getcwd()
data_file = wd.split('/')[-1]
data_file = wd + '/' + data_file
with open(data_file,'wb') as f:
	save={
		'image': images
	}
	pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
	f.close()
statinfo = os.stat(data_file)
file_size = float(statinfo.st_size)/1000
print('Compressed data size: %0.1fkB' % file_size)
