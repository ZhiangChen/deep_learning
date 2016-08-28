/// kinect2depth
/// Zhiang Chen, Wyatt Newman Aug 2016
/// Receive the pointcloud data from "kinect2/qhd/points", and then project the points in the box filter to a depth image. 
/// Publish the depth image to "depth_image" with 10hz.

/*
 * The MIT License (MIT)
 *  Copyright (c) 2016 Zhiang Chen, Wyatt Newman
 */

#include <ros/ros.h> 
#include <stdlib.h>
#include <math.h>
#include <sensor_msgs/PointCloud2.h> 
#include <sensor_msgs/Image.h>
#include <pcl_ros/point_cloud.h> 
#include <pcl/conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common_headers.h>
#include <pcl-1.7/pcl/point_cloud.h>
#include <pcl-1.7/pcl/PCLHeader.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h> 
#include <pcl_utils/pcl_utils.h>  
#include <iostream>
#include <fstream>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;

#define DISPLAY
void loadParameters(ros::NodeHandle nh);
sensor_msgs::Image project(pcl::PointCloud<pcl::PointXYZRGB>::Ptr  box_ptr, int npts_cloud);
void box_filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr  inputCloud, Eigen::Vector3f pt_min, Eigen::Vector3f pt_max, vector<int> &indices);
/*********  Hyperparameters  **********/
// Plane Paramters 
float _A = 0.00081042;
float _B = 0.503923;
float _C = 0.863748;
float _D = -0.800717;
float _cX = -0.14675;
float _cY = 0.109414;
float _cZ = 0.863329;
// Box Filter Parameters
float _bnX = -0.06;
float _bnY = -0.11;
float _bnZ = -0.015;
float _bmX = 0.115;
float _bmY = 0.08;
float _bmZ = 0.1;
// Projection Parameters
float _mDis = 0.92;  // darkest, float!
float _nDis = 0.77; // brightest, float!
float _Nv = 120;
float _Nu = 120;
float _focal_len = 200.0; //220.0 may be good? float!
/*************************************/
// 500x250x250

pcl::PointCloud<pcl::PointXYZRGB>::Ptr g_kinect_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
bool g_got_data = false;
void kinectCallback(const sensor_msgs::PointCloud2ConstPtr &cloud)
{
	g_got_data = true;
	pcl::fromROSMsg(*cloud, *g_kinect_ptr);
}


int main(int argc, char** argv) 
{
    ros::init(argc, argv, "kinect2depth2"); //node name
    ros::NodeHandle nh;
    PclUtils pclUtils(&nh);
    loadParameters(nh);
// subscribers
    ros::Subscriber sub = nh.subscribe("kinect2/qhd/points", 1, kinectCallback);

// publishers
    ros::Publisher pub_depth_image = nh.advertise<sensor_msgs::Image> ("/depth_image",1);
    ros::Publisher pub_kinect = nh.advertise<sensor_msgs::PointCloud2> ("/kinect", 1);
    ros::Publisher pub_tf_kinect = nh.advertise<sensor_msgs::PointCloud2> ("/tf_kinect", 1);
    ros::Publisher pub_tf_box = nh.advertise<sensor_msgs::PointCloud2> ("/tf_box", 1);
    ros::Publisher pub_box = nh.advertise<sensor_msgs::PointCloud2> ("/box", 1);

// pcl pointcloud    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr kinect_ptr(new pcl::PointCloud<pcl::PointXYZRGB>); // pointer to the pointcloud wrt kinect coords
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tf_kinect_ptr(new pcl::PointCloud<pcl::PointXYZRGB>); // pointer to the pointcloud wrt plate coords
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tf_box_ptr(new pcl::PointCloud<pcl::PointXYZRGB>); // pointer to the interesting points (in the box) wrt plate coords
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr box_ptr(new pcl::PointCloud<pcl::PointXYZRGB>); // pointer to the interesting points (in the box) wrt camera coords

    // pointer to downsampled pointcloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tf_ds_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tf_ds_box_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

// sensor msg pointcloud
    sensor_msgs::PointCloud2 ros_kinect, ros_box, ros_tf_kinect, ros_tf_box, ros_tf_ds, ros_tf_ds_box; 

// sensor msg depth image
	  sensor_msgs::Image depth_image;    

// The transform from camera to plane is hard-coded.
    Eigen::Affine3f A_plane_wrt_camera;
    Eigen::Vector4f plane_parameters;
    Eigen::Vector3f plane_centroid3f;

    plane_parameters<<_A, _B, _C, _D;
    plane_centroid3f<<_cX, _cY, _cZ;
    
    A_plane_wrt_camera = pclUtils.make_affine_from_plane_params(plane_parameters,plane_centroid3f);

// Image projection variables
	  Eigen::Vector3f cloud_pt;
    double x,y,z;
  	double v,vc,u,uc;
  	int i,j;
  	uc = _Nu/2.0;
  	vc = _Nv/2.0;
  	uchar gray_level;
  	double r;
  	int npts_cloud;
  	cv::Mat image(_Nu,_Nv,CV_8U,cv::Scalar(0));

// wait for kinect data
    ROS_INFO("Initialized!");
  	while(!g_got_data)
  	{
  		ros::spinOnce();
  		ros::Duration(1.0).sleep();
  	}
  	ROS_INFO("Got Kinect Data!");

// get and publish depth image
    vector<int> indices; // indices of interesting pixels
    // parameters for box filter wrt plane coords
    Eigen::Vector3f box_pt_min,box_pt_max;
    box_pt_min<<_bnX,_bnY,_bnZ;
    box_pt_max<<_bmX,_bmY,_bmZ;
  	while(ros::ok())
  	{
  		kinect_ptr = g_kinect_ptr;
  		// transform to plane coords
    	pcl::transformPointCloud(*kinect_ptr, *tf_kinect_ptr, A_plane_wrt_camera.inverse());
    	// box filter
    	box_filter(tf_kinect_ptr, box_pt_min, box_pt_max, indices);
    	pcl::copyPointCloud(*tf_kinect_ptr, indices, *tf_box_ptr);
    	pcl::copyPointCloud(*kinect_ptr, indices, *box_ptr);
    	// get depth image
    	int npts_cloud = indices.size();
    	depth_image = project(box_ptr, npts_cloud);
    	// publish depth image
    	pub_depth_image.publish(depth_image);
    	// refresh data
    	g_got_data = false;
    	while(!g_got_data && ros::ok())
    	{
    		ros::spinOnce();
    		ros::Duration(0.5).sleep();
    	}

#ifdef DISPLAY
    	pcl::toROSMsg(*tf_kinect_ptr, ros_tf_kinect);
    	pcl::toROSMsg(*kinect_ptr, ros_kinect);
    	pcl::toROSMsg(*tf_box_ptr, ros_tf_box);
    	pcl::toROSMsg(*box_ptr, ros_box);

		ros_kinect.header.frame_id = "camera";
		ros_tf_kinect.header.frame_id = "plane";
		ros_tf_box.header.frame_id = "plane";
		ros_box.header.frame_id = "camera";
		
		pub_kinect.publish(ros_kinect);
		pub_tf_kinect.publish(ros_tf_kinect);
		pub_tf_box.publish(ros_tf_box);
		pub_box.publish(ros_box);
#endif

  	}
  	return 0;
}

sensor_msgs::Image project(pcl::PointCloud<pcl::PointXYZRGB>::Ptr  box_ptr, int npts_cloud)
{
	Eigen::Vector3f cloud_pt;
    double x,y,z;
  	double v,vc,u,uc;
  	int i,j;
  	uc = _Nu/2.0;
  	vc = _Nv/2.0;
  	uchar gray_level;
  	double r;
  	cv::Mat image(_Nu,_Nv,CV_8U,cv::Scalar(0));
  	sensor_msgs::Image depth_image;

	// project to a depth image
    for (int ipt = 0;ipt<npts_cloud;ipt++) // i-th point
    {
	    cloud_pt = box_ptr->points[ipt].getVector3fMap();
	    z = cloud_pt[2];
	    y = cloud_pt[1];
	    x = cloud_pt[0];
	    if ((z==z)&&(x==x)&&(y==y)) 
	    { 
	       u = uc + _focal_len*x/z;
	       i = round(u);
	       v = vc + _focal_len*y/z;
	       j = round(v);
	       if ((i>=0)&&(i<_Nu)&&(j>=0)&&(j<_Nv)) 
	       {
	           // convert z to an intensity:
	           r = sqrt(z*z+y*y+x*x);
	           if (r>_mDis) gray_level=0;
	           else if (r<_nDis) gray_level=0;
	           else 
	           {
	           		gray_level = (uchar) (255*(_mDis-r)/(_mDis-_nDis));
	           }
	           image.at<uchar>(j,i)= gray_level;
	       	}

	   	}
	}
	// convert opencv image to ros image
	cv_bridge::CvImage cvi;
	cvi.header.stamp = ros::Time::now();
	cvi.header.frame_id = "camera";
	cvi.encoding = "mono8";
	cvi.image = image;
	cvi.toImageMsg(depth_image);

	return depth_image;
}

void box_filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr  inputCloud, Eigen::Vector3f pt_min, Eigen::Vector3f pt_max, 
                vector<int> &indices)  {
    int npts = inputCloud->points.size();
    Eigen::Vector3f pt;
    indices.clear();
    for (int i = 0; i < npts; ++i) {
        pt = inputCloud->points[i].getVector3fMap();
        //cout<<"pt: "<<pt.transpose()<<endl;
        //check if in the box:
        if ((pt[0]>pt_min[0])&&(pt[0]<pt_max[0])&&(pt[1]>pt_min[1])&&(pt[1]<pt_max[1])&&(pt[2]>pt_min[2])&&(pt[2]<pt_max[2])) { 
            //passed box-crop test; include this point
               indices.push_back(i);
        }
    }
    int n_extracted = indices.size();
    cout << " number of points within box = " << n_extracted << endl;    
    
}

void loadParameters(ros::NodeHandle nh)
{
    nh.getParam("A",_A);
    nh.getParam("B",_B);
    nh.getParam("C",_C);
    nh.getParam("D",_D);
    nh.getParam("cX",_cX);
    nh.getParam("cY",_cY);
    nh.getParam("cZ",_cZ);
    nh.getParam("bnX",_bnX);
    nh.getParam("bnY",_bnY);
    nh.getParam("bnZ",_bnZ);
    nh.getParam("bmX",_bmX);
    nh.getParam("bmY",_bmY);
    nh.getParam("bmZ",_bmZ);
    nh.getParam("Nv",_Nv);
    nh.getParam("Nu",_Nu);
    nh.getParam("focal_len",_focal_len);
    nh.getParam("mDis",_mDis);
    nh.getParam("nDis",_nDis);
}