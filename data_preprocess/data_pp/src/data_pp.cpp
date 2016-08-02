/// data_pp.cpp
/// Zhiang Chen, 7/2016
/// Read pcd files in the current directory, extract interesting points and project to depth images.

#include <ros/ros.h> 
#include <stdlib.h>
#include <math.h>
#include <sensor_msgs/PointCloud2.h> 
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

void box_filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr  inputCloud, Eigen::Vector3f pt_min, Eigen::Vector3f pt_max, vector<int> &indices);
void projection_method1(pcl::PointCloud<pcl::PointXYZRGB>::Ptr  box_ptr, int npts_cloud, string fname);
void projection_method2(pcl::PointCloud<pcl::PointXYZRGB>::Ptr  box_ptr, int npts_cloud, string fname, double resolution);

using namespace std;

#define DISPLAY

/*********  Hyperparameters  **********/
// Plane Paramters 
#define A 0.00081042
#define B 0.503923
#define C 0.863748
#define D -0.800717
#define cX -0.14675
#define cY 0.109414
#define cZ 0.863329
// Box Filter Parameters
#define bnX -0.06
#define bnY -0.11
#define bnZ -0.015
#define bmX 0.115
#define bmY 0.08
#define bmZ 0.1
// Projection Parameters
#define mDis 0.92  // darkest, float!
#define nDis 0.77 // brightest, float!
#define Nv 120
#define Nu 120
#define focal_len 200.0 //220.0 may be good? float!
/*************************************/
// 500x250x250

int main(int argc, char** argv) {
    ros::init(argc, argv, "data_pp"); //node name
    ros::NodeHandle nh;
    PclUtils pclUtils(&nh);

// publishers
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

// The transform from camera to plane is hard-coded.
    Eigen::Affine3f A_plane_wrt_camera;
    Eigen::Vector4f plane_parameters;
    Eigen::Vector3f plane_centroid3f;

    plane_parameters<<A, B, C, D;
    plane_centroid3f<<cX, cY, cZ;
    
    A_plane_wrt_camera = pclUtils.make_affine_from_plane_params(plane_parameters,plane_centroid3f);

// Image projection variables
	Eigen::Vector3f cloud_pt;
    double x,y,z;
  	double v,vc,u,uc;
  	int i,j;
  	uc = Nu/2.0;
  	vc = Nv/2.0;
  	uchar gray_level;
  	double r;
  	int npts_cloud;
  	cv::Mat image(Nu,Nv,CV_8U,cv::Scalar(0));

// read the list of names
    ifstream file;
    file.open("name_lists");
    string name;
    if (!file.is_open())
    {
    	ROS_ERROR("WRONG DIRECTORY! FAILED TO FIND THE FILE!!");
    	return -1;
    }
    vector<string> names;
    names.clear();
    while (!file.eof()) 
    {
    	// load file
    	file >> name;
    	if (name!="name_lists")
    	{
    		names.push_back(name);
    	}
    	cout<<name<<endl;
    }

// get depth images
    string fname; // image's name
    vector<int> indices; // indices of interesting pixels
    int num = names.size(); // the number of images 
    ROS_INFO("There are %d images in total.", num);
    // parameters for box filter wrt plane coords
    Eigen::Vector3f box_pt_min,box_pt_max;
    box_pt_min<<bnX,bnY,bnZ;
    box_pt_max<<bmX,bmY,bmZ;

    for (int ipic=0; ipic<num; ipic++) // i-th picture
    {
    	// load image by name
    	fname = names[ipic];
    	if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (fname, *kinect_ptr) == -1)
    	{
        	ROS_ERROR("Couldn't read file \n");
        	return -1;
    	}
    	cout<<"............................................................."<<endl;
    	ROS_INFO_STREAM("Projecting file: "<<fname);
    	// transform to plane coords
    	pcl::transformPointCloud(*kinect_ptr, *tf_kinect_ptr, A_plane_wrt_camera.inverse());
    	// box filter
    	box_filter(tf_kinect_ptr, box_pt_min, box_pt_max, indices);
    	pcl::copyPointCloud(*tf_kinect_ptr, indices, *tf_box_ptr);
    	pcl::copyPointCloud(*kinect_ptr, indices, *box_ptr);

    	int npts_cloud = indices.size();
    	projection_method1(box_ptr, npts_cloud, fname);
    	cout<<"Finished "<<std::fixed<<std::setprecision(1)<< double(ipic+1)/num*100 <<"%"<<endl;

#ifdef DISPLAY
    	pcl::toROSMsg(*tf_kinect_ptr, ros_tf_kinect);
    	pcl::toROSMsg(*kinect_ptr, ros_kinect);
    	pcl::toROSMsg(*tf_box_ptr, ros_tf_box);
    	pcl::toROSMsg(*box_ptr, ros_box);

    	while(ros::ok())
    	{
    		ros_kinect.header.frame_id = "camera";
    		ros_tf_kinect.header.frame_id = "plane";
    		ros_tf_box.header.frame_id = "plane";
    		ros_box.header.frame_id = "camera";
    		
    		pub_kinect.publish(ros_kinect);
    		pub_tf_kinect.publish(ros_tf_kinect);
    		pub_tf_box.publish(ros_tf_box);
    		pub_box.publish(ros_box);
    		ros::Duration(1.0).sleep(); 
    	}
#endif

    }
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

void projection_method1(pcl::PointCloud<pcl::PointXYZRGB>::Ptr  box_ptr, int npts_cloud, string fname)
{
	Eigen::Vector3f cloud_pt;
    double x,y,z;
  	double v,vc,u,uc;
  	int i,j;
  	uc = Nu/2.0;
  	vc = Nv/2.0;
  	uchar gray_level;
  	double r;
  	cv::Mat image(Nu,Nv,CV_8U,cv::Scalar(0));

	// project to a depth image
    for (int ipt = 0;ipt<npts_cloud;ipt++) // i-th point
    {
	    cloud_pt = box_ptr->points[ipt].getVector3fMap();
	    z = cloud_pt[2];
	    y = cloud_pt[1];
	    x = cloud_pt[0];
	    if ((z==z)&&(x==x)&&(y==y)) 
	    { 
	       u = uc + focal_len*x/z;
	       i = round(u);
	       v = vc + focal_len*y/z;
	       j = round(v);
	       if ((i>=0)&&(i<Nu)&&(j>=0)&&(j<Nv)) 
	       {
	           // convert z to an intensity:
	           r = sqrt(z*z+y*y+x*x);
	           if (r>mDis) gray_level=0;
	           else if (r<nDis) gray_level=0;
	           else 
	           {
	           		gray_level = (uchar) (255*(mDis-r)/(mDis-nDis));
	           }
	           image.at<uchar>(j,i)= gray_level;
	       	}

	   	}
	}

	// save image
	size_t lastindex = fname.find_last_of("."); 
	string newname = fname.substr(0, lastindex); 
	newname += ".bmp";
	cv::imwrite(newname, image);  
	ROS_INFO_STREAM("Saved file: "<<newname);

}

/*
void projection_method2(pcl::PointCloud<pcl::PointXYZRGB>::Ptr  box_ptr, int npts_cloud, string fname, double resolution)
*/
