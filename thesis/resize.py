import cv2
import numpy as np
import deepdish as dd
import os
import matplotlib.pyplot as plt

file_name = 'depth_data.h5'
width = 48
height = 48
mean = 0.15

save = dd.io.load(file_name)

train_objects = save['train_objects']
train_orientations = save['train_orientations']
train_values = save['train_values']
valid_objects = save['valid_objects']
valid_orientations = save['valid_orientations']
valid_values = save['valid_values']
test_objects = save['test_objects']
test_orientations = save['test_orientations']
test_values = save['test_values']
value2object = save['value2object']
object2value = save['object2value']
del save


print('training dataset', train_objects.shape, train_orientations.shape, train_values.shape)
print('validation dataset', valid_objects.shape, valid_orientations.shape, valid_values.shape)
print('testing dataset', test_objects.shape, test_orientations.shape, test_values.shape)

def image_resize(images, average=True):
	resized_images = np.asarray([cv2.resize(image,(height,width)) for image in images])
	if average:
		return resized_images - mean
	else:
		return resized_images

train_values = image_resize(train_values)
test_values = image_resize(test_values)
valid_values = image_resize(valid_values)
print('*'*40)
print('training dataset', train_objects.shape, train_orientations.shape, train_values.shape)
print('validation dataset', valid_objects.shape, valid_orientations.shape, valid_values.shape)
print('testing dataset', test_objects.shape, test_orientations.shape, test_values.shape)

data_file = 'resized_depth_data2'
save={
    'train_orientations':train_orientations,
    'valid_orientations':valid_orientations,
    'test_orientations':test_orientations,
    'train_objects':train_objects,
    'valid_objects':valid_objects,
    'test_objects':test_objects,
    'train_values':train_values,
    'valid_values':valid_values,
    'test_values':test_values,
    'object2value':object2value,
    'value2object':value2object
}

dd.io.save('resized_depth_data2.h5', save, compression=None)