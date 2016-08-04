#!/usr/bin/env python3
'''
Zhiang Chen
Aug,2016
'''

"An example of importing depth data"

from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
wd = input("Input the path of depth_data: ")
file_name = wd+'/depth_data'
with open(file_name, 'rb') as f:
    save = pickle.load(f)
    dataset = save['dataset']
    names = save['names']
    orientations = save['orientations']
    del save
#print(dataset.shape)
image1 = dataset[0,:,:]
plt.imshow(image1,cmap='Greys_r')
plt.show()
