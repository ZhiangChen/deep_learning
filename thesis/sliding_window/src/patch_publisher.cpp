/// patch publisher
/// Zhiang Chen, March 2017
/// publish patch to topic 'patch'
/*
 * The MIT License (MIT)
 *  Copyright (c) 2017 Zhiang Chen
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
bool loadParameters(ros::NodeHandle nh);
void box_filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr  inputCloud, Eigen::Vector3f pt_min, Eigen::Vector3f pt_max, vector<int> &indices);
/*********  Hyperparameters  **********/
// Plane Paramters 
float _A = -0.0527697168;
float _B = 0.524196565;
float _C = 0.849960804;
float _D = -0.871602178;
float _cX = 0.0519640781;
float _cY = 0.305142462;
float _cZ = 0.845000029;
// Box Filter Parameters
float _bnX = -0.09;
float _bnY = -0.09;
float _bnZ = -0.015;
float _bmX = 0.09;
float _bmY = 0.09;
float _bmZ = 0.1;
// Patch Parameters
float _xl = -0.04552;
float _xr = 0.06415;
float _yu = 0.069428;
float _yd = -0.04674;

pcl::PointCloud<pcl::PointXYZRGB>::Ptr g_kinect_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
bool g_got_data = false;
void kinectCallback(const sensor_msgs::PointCloud2ConstPtr &cloud)
{
	g_got_data = true;
	pcl::fromROSMsg(*cloud, *g_kinect_ptr);
}

int main(int argc, char** argv) 
{
    ros::init(argc, argv, "patch publisher"); //node name
    ros::NodeHandle nh;
    PclUtils pclUtils(&nh);
    while(!loadParameters(nh))
    {
    	ROS_ERROR("LOAD PARAMETERS!");
    	ros::Duration(1.0).sleep();
    } 

    ros::Subscriber sub = nh.subscribe("kinect2/qhd/points", 1, kinectCallback);

    ros::Publisher pub_kinect = nh.advertise<sensor_msgs::PointCloud2> ("/kinect", 1);
    ros::Publisher pub_tf_kinect = nh.advertise<sensor_msgs::PointCloud2> ("/tf_kinect", 1);
    ros::Publisher pub_tf_patch = nh.advertise<sensor_msgs::PointCloud2> ("/tf_patch", 1);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr kinect_ptr(new pcl::PointCloud<pcl::PointXYZRGB>); // pointer to the pointcloud wrt kinect coords
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tf_kinect_ptr(new pcl::PointCloud<pcl::PointXYZRGB>); // pointer to the pointcloud wrt plate coords
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tf_patch_ptr(new pcl::PointCloud<pcl::PointXYZRGB>); // pointer to the interesting points (in the box) wrt plate coords

    sensor_msgs::PointCloud2 ros_kinect, ros_tf_kinect, ros_tf_patch; 

    Eigen::Affine3f A_plane_wrt_camera;
    Eigen::Vector4f plane_parameters;
    Eigen::Vector3f plane_centroid3f;

    plane_parameters<<_A, _B, _C, _D;
    plane_centroid3f<<_cX, _cY, _cZ;

    A_plane_wrt_camera = pclUtils.make_affine_from_plane_params(plane_parameters,plane_centroid3f);

    ROS_INFO("Initialized!");
  	while(!g_got_data)
  	{
  		ros::spinOnce();
  		ros::Duration(1.0).sleep();
  	}
  	ROS_INFO("Got Kinect Data!");

    vector<int> indices; // indices of interesting pixels
    // parameters for box filter wrt plane coords
    Eigen::Vector3f box_pt_min,box_pt_max;
    box_pt_min<<(_xl+_bnX), (_yd+_bnY), _bnZ;
    box_pt_max<<(_xr+bmX), (_yu+bmX), _bmZ;
    int point_nm;

    while(ros::ok())
    {
  		kinect_ptr = g_kinect_ptr;
  		// transform to plane coords
    	pcl::transformPointCloud(*kinect_ptr, *tf_kinect_ptr, A_plane_wrt_camera.inverse());

    	box_filter(tf_kinect_ptr, box_pt_min, box_pt_max, indices);
    	pcl::copyPointCloud(*tf_kinect_ptr, indices, *tf_patch_ptr);

    	g_got_data = false;
    	while(!g_got_data && ros::ok())
    	{
    		ros::spinOnce();
    		ros::Duration(0.5).sleep();
    	}

#ifdef DISPLAY
    	pcl::toROSMsg(*tf_kinect_ptr, ros_tf_kinect);
    	pcl::toROSMsg(*kinect_ptr, ros_kinect);
    	pcl::toROSMsg(*tf_patch_ptr, ros_tf_patch);

		ros_kinect.header.frame_id = "camera";
		ros_tf_kinect.header.frame_id = "plane";
		ros_tf_patch.header.frame_id = "plane";
		
		pub_kinect.publish(ros_kinect);
		pub_tf_kinect.publish(ros_tf_kinect);
		pub_tf_patch.publish(ros_tf_patch);

#endif
    }
    return 0;

}

bool loadParameters(ros::NodeHandle nh)
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
    nh.getParam("xl",_xl);
    nh.getParam("xr",_xr);
    nh.getParam("yu",_yu);
    nh.getParam("yd",_yd);

    return true;
}

void box_filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr  inputCloud, Eigen::Vector3f pt_min, Eigen::Vector3f pt_max, vector<int> &indices)  
{
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
    cout << " number of points within big box = " << n_extracted << endl;    
    
}