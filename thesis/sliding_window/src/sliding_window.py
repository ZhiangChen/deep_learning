#! /usr/bin/python
'''
Sliding Window for Object Detection
Zhiang Chen, March 2017

The MIT License (MIT)
Copyright (c) 2017 Zhiang Chen
'''

# read bnZ bmZ
# parallel processing: https://docs.python.org/2/library/multiprocessing.html
# visualization::MarkerArray: text for class; arrow for angle
# f: get boxes
# f: get depth_images ##parallel
# f: evaluate ##np.append
# f: filter out those below threshold
# f: get the best as seeds and compute the mean of same classes in certain range (angle as well)
# f: publish class labels and accuracy
# f: publish angle marker

import rospy
import roslib
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from orthaffine import OrthAffine as OA
from evaluator import *
from math import *


theta = 30.0/180*pi
x_step = 0.005
y_step = 0.005

class SlidingWindow():
	def __init__(self):
		self.sub1 = rospy.Subscriber('/tf_patch', PointCloud2, self.callback, queue_size=1)
		self.pub1 = rospy.Publisher('objects', MarkerArray, queue_size=1)
		self.pub2 = rospy.Publisher('orientations', MarkerArray, queue_size=1)
		self.xl = rospy.get_param('xl')
		self.xr = rospy.get_param('xr')
		self.yu = rospy.get_param('yu')
		self.yd = rospy.get_param('yd')
		self.bnX = rospy.get_param('bnX')
		self.bmX = rospy.get_param('bmX')
		self.bnY = rospy.get_param('bnY')
		self.bmY = rospy.get_param('bmY')		
		self.bnZ = rospy.get_param('bnZ')
		self.bmZ = rospy.get_param('bmZ')
		self.image_size = rospy.get_param('image_size')
		self.centers = self.get_box_centers()
		self.oa = OA(theta)
		self.ev = evaluator()
		rospy.loginfo("Sliding Window Initialized!")
		self.i=0;

	def get_box_centers(self):
		x_centers = np.arange(self.xl, self.xr, x_step)
		y_centers = np.arange(self.yd, self.yu, y_step)
		x_centers, y_centers = np.meshgrid(x_centers, y_centers)
		x_centers = x_centers.reshape(x_centers.shape[0],x_centers.shape[1],1)
		y_centers = y_centers.reshape(y_centers.shape[0],y_centers.shape[1],1)
		centers = np.concatenate((x_centers,y_centers),axis=2).reshape(-1,2)
		return centers

	def box_filter(self,center):
		xl = center[0] + self.bnX
		xr = center[0] + self.bmX
		yu = center[1] + self.bmY
		yd = center[1] + self.bnY
		#print('center',xl,xr,yu,yd)
		# filter
		box = self.patch_points[(self.patch_points[:,0]>xl) & (self.patch_points[:,0]<xr) \
		& (self.patch_points[:,1]<yu) & (self.patch_points[:,1]>yd)\
		& (self.patch_points[:,2]>self.bnZ) & (self.patch_points[:,2]<self.bmZ)]
		#print('box shape',box.shape)
		image = self.get_image(box, center)
		return image

	def get_image(self, box, center):
		xl = center[0] + self.bnX
		xr = center[0] + self.bmX
		yu = center[1] + self.bmY
		yd = center[1] + self.bnY		
		self.oa.readpoints(box)
		self.oa.affine()
		self.oa.interpolate(theta,bnX=xl,bmX=xr,bnY=yd,bmY=yu)
		image_array = self.oa.project()
		#self.i = self.i+1
		#self.oa.saveimage(str(self.i)+'.png')
		return image_array

	def single_object_filter(self,score=0.80):
		# remove empty plate
		self.results = self.results[self.results[:,1]>0.5]
		#print(self.results.shape)
		# remove low scores
		self.results = self.results[self.results[:,0]>score]
		#print(self.results.shape)
		objects = self.results[:,1].tolist()
		objects = [int(obj) for obj in objects]
		objects = set(objects) # remove the duplicates in the list
		ct = list()
		agl = list()
		clas = list()
		for obj in objects:
			pts = self.results[(self.results[:,1]>obj) & (self.results[:,1]<(obj+0.5))]
			centers = pts[:,3:5]
			centers = np.repeat(centers,2,axis=0)
			#print('centers',centers)
			#print(centers.shape)
			center = np.mean(centers,axis=1)
			#print('center',center)
			angles = pts[:,2]
			#print(angles)
			angle = np.mean(angles)
			ct.append(center)
			agl.append(angle)
			clas.append(obj)
		return np.asarray(ct), agl, clas

	def callback(self, patch_points):
		generator = pc2.read_points(patch_points, skip_nans=True, field_names=("x", "y", "z"))
		pts = list()
		for i in generator:
			pts.append(i)
		self.patch_points = np.asarray(pts)
		centers = self.centers.tolist()
		#print(centers)
		rospy.loginfo("The number of centers: %d" % len(centers))
		images = [self.box_filter(center) for center in centers]
		images = np.asarray(images).reshape(-1,self.image_size,self.image_size,1).astype(np.float32)
		"ADD SMALL DATA"
		nm_images = images.shape[0]
		images = np.concatenate((images,small_data),axis=0)
		classes, scores, angles = self.ev.evaluate(images)
		classes = classes[:nm_images,:]
		scores = scores[:nm_images,:]
		angles = angles[:nm_images,:]
		classes_ = [value2name[int(value[0])] for value in classes.tolist()]
		classes_ = [name2string[name] for name in classes_]
		print(classes_)
		print(scores)
		self.results = np.concatenate((scores,classes,angles,self.centers[:,0].reshape(-1,1),self.centers[:,1].reshape(-1,1)),axis=1)
		centers, angles, objects = self.single_object_filter()
		if len(objects)!=0:
			print('classes:', name2string[value2name[objects[0]]])
			self.publish_classes(objects,centers)
			self.publish_angles(angles,centers)


	def publish_classes(self, classes, centers):
		markerArray = MarkerArray()
		for index, clas in enumerate(classes):
			marker = Marker()
			marker.header.frame_id = 'plane'
			marker.type = Marker.TEXT_VIEW_FACING
			marker.action = marker.ADD
			marker.scale.x = 0.05
			marker.scale.y = 0.05
			marker.scale.z = 0.05
			marker.color.a = 1.0
			marker.color.r = 1.0
			marker.color.g = 1.0
			marker.color.b = 0.0
			marker.pose.orientation.w = 1.0
			marker.pose.position.x = centers[index,0]
			marker.pose.position.y = centers[index,1]
			marker.pose.position.z = self.bmZ + 0.05
			marker.text = name2string[value2name[clas]]
			markerArray.markers.append(marker)
		self.pub1.publish(markerArray)

	def publish_angles(self, angles, centers):
		markerArray = MarkerArray()
		for index, angle in enumerate(angles):
			marker = Marker()
			marker.header.frame_id = 'plane'
			marker.type = marker.TEXT_VIEW_FACING
			marker.action = marker.ADD
			marker.scale.x = 0.05
			marker.scale.y = 0.05
			marker.scale.z = 0.05
			marker.color.a = 1.0
			marker.color.r = 1.0
			marker.color.g = 0.0
			marker.color.b = 0.0
			marker.pose.orientation.w = 1.0
			marker.pose.position.x = centers[index,0]
			marker.pose.position.y = centers[index,1]
			marker.pose.position.z = self.bmZ 
			marker.text = 'angle: '+str(angles[index])
			markerArray.markers.append(marker)
		self.pub2.publish(markerArray)

if __name__ == '__main__':
    rospy.init_node('sliding_window',anonymous=True)
    sw = SlidingWindow()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS node sliding window")
    session.close()
    print("Shutting down ROS node sliding window")


