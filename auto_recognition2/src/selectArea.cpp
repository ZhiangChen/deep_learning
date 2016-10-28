/// selectArea
/// Zhiang Chen, Aug 2016
/// Require users to select an area using "PublishSelectedPoint" in RViz, then store parameters in yaml file.

/*
 * The MIT License (MIT)
 *  Copyright (c) 2016 Zhiang Chen
 */

#include <ros/ros.h> 
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
using namespace std;
void box_filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr  inputCloud, Eigen::Vector3f pt_min, Eigen::Vector3f pt_max, vector<int> &indices);

#define box_x 0.19
#define box_y 0.19
#define bnz -0.015 
#define bmz 0.1
#define h_proction 0.17
#define focal_len 210.0
#define Nv 200
#define Nu 200
#define image_size 100

pcl::PointCloud<pcl::PointXYZRGB>::Ptr g_kinect_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
bool g_got_data = false;
void kinectCallback(const sensor_msgs::PointCloud2ConstPtr &cloud)
{
	g_got_data = true;
	pcl::fromROSMsg(*cloud, *g_kinect_ptr);
}

int main(int argc, char** argv) 
{
    ros::init(argc, argv, "selectArea"); //node name
    ros::NodeHandle nh;
    PclUtils pclUtils(&nh);

    ros::Subscriber sub = nh.subscribe("kinect2/qhd/points", 1, kinectCallback);
// publishers
    ros::Publisher pub_depth_image = nh.advertise<sensor_msgs::Image> ("/depth_image",1);
    ros::Publisher pub_kinect = nh.advertise<sensor_msgs::PointCloud2> ("/kinect", 1);
    ros::Publisher pub_tf_kinect = nh.advertise<sensor_msgs::PointCloud2> ("/tf_kinect", 1);
    ros::Publisher pub_tf_box = nh.advertise<sensor_msgs::PointCloud2> ("/tf_box", 1);
    ros::Publisher pub_box = nh.advertise<sensor_msgs::PointCloud2> ("/box", 1);

// pcl pointcloud    
    pcl::PointCloud<pcl::PointXYZ>::Ptr selected_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    int selected_nm;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr kinect_ptr(new pcl::PointCloud<pcl::PointXYZRGB>); // pointer to the pointcloud wrt kinect coords
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tf_kinect_ptr(new pcl::PointCloud<pcl::PointXYZRGB>); // pointer to the pointcloud wrt plate coords
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tf_box_ptr(new pcl::PointCloud<pcl::PointXYZRGB>); // pointer to the interesting points (in the box) wrt plate coords
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr box_ptr(new pcl::PointCloud<pcl::PointXYZRGB>); // pointer to the interesting points (in the box) wrt camera coords

// sensor msg pointcloud
    sensor_msgs::PointCloud2 ros_kinect, ros_box, ros_tf_kinect, ros_tf_box, ros_tf_ds, ros_tf_ds_box; 

// parameters
    float curvature;
    Eigen::Vector4f plane_parameters; // wrt camera
    Eigen::Vector3f plane_centroid; // wrt camera (x,y,z)
    Eigen::Vector4f plane_centroid4; // wrt camera (x,y,z,1)
    Eigen::Vector4f plane_centroid_wrt_plane; // wrt plane (x,y,z,1)
    double dist_centroid; // wrt camera, the distance between centroid and camera
    Eigen::Affine3f A_plane_wrt_camera;
    Eigen::Vector3f boxn; // wrt plane
    Eigen::Vector3f boxm; // wrt plane
    vector<int> indices;
    double mDis; // projection parameter, darkest
    double nDis; // projection parameter, brightest
    Eigen::Vector3f top_left,bottom_right; // wrt camera
    double u,v,uc,vc;
    double x,y,z;
    int i,j,i2,j2;

    while(!g_got_data)
	{
		ros::spinOnce();
		ros::Duration(0.5).sleep();
	}
	ROS_INFO("Got kinect data!");
	g_got_data = false;
	kinect_ptr = g_kinect_ptr;
	ROS_INFO("Select a centroid!");
	bool got_param = false;

    while(ros::ok())
    {
    	if (pclUtils.got_selected_points())
    	{
    		pclUtils.get_copy_selected_points(selected_ptr); 
    		//selected_nm = selected_ptr->points.size();
    		//ROS_INFO("The number of selected points is: %d", selected_nm);
    		plane_centroid = pclUtils.compute_centroid(selected_ptr);
    		pclUtils.reset_got_selected_points();
    		ROS_INFO("Select a patch!");
    		while(!pclUtils.got_selected_points())
			{
				ros::spinOnce();
				ros::Duration(0.5).sleep();
			}
			pclUtils.reset_got_selected_points();
			pclUtils.get_copy_selected_points(selected_ptr); 
			pcl::computePointNormal(*selected_ptr, plane_parameters, curvature); 
    		dist_centroid = plane_centroid.norm();
    		mDis = dist_centroid + h_proction/2 ;
    		nDis = dist_centroid - h_proction/2 ;
    		plane_centroid4[0] = plane_centroid[0];
    		plane_centroid4[1] = plane_centroid[1];
    		plane_centroid4[2] = plane_centroid[2];
    		plane_centroid4[3] = 1.0;
    		A_plane_wrt_camera = pclUtils.make_affine_from_plane_params(plane_parameters,plane_centroid);
    		pcl::transformPointCloud(*kinect_ptr, *tf_kinect_ptr, A_plane_wrt_camera.inverse());
    		plane_centroid_wrt_plane = A_plane_wrt_camera.inverse()*plane_centroid4;
    		boxn[0] = plane_centroid_wrt_plane[0] - box_x/2;
    		boxn[1] = plane_centroid_wrt_plane[1] - box_y/2;
    		boxn[2] = bnz;
    		boxm[0] = plane_centroid_wrt_plane[0] + box_x/2;
    		boxm[1] = plane_centroid_wrt_plane[1] + box_y/2;
    		boxm[2] = bmz;
    		// box filter
    		box_filter(tf_kinect_ptr, boxn, boxm, indices);
    		pcl::copyPointCloud(*tf_kinect_ptr, indices, *tf_box_ptr);
    		pcl::copyPointCloud(*kinect_ptr, indices, *box_ptr);
    		// top-left pixel of cropped image
    		top_left[0] = plane_centroid_wrt_plane[0] - box_x/2;
    		top_left[1] = plane_centroid_wrt_plane[1] + box_y/2;
    		top_left[2] = 0;
    		top_left = A_plane_wrt_camera*top_left;
    		z = top_left[2];
	    	y = top_left[1];
	    	x = top_left[0];
    		uc = Nu/2.0;
  			vc = Nv/2.0;
    		u = uc + focal_len*x/z;
	        i = round(u)+1;
	        v = vc + focal_len*y/z;
	        j = round(v)+1;

            bottom_right[0] = plane_centroid_wrt_plane[0] + box_x/2;
            bottom_right[1] = plane_centroid_wrt_plane[1] - box_y/2;
            bottom_right[2] = 0;
            bottom_right = A_plane_wrt_camera*bottom_right;
            z = bottom_right[2];
            y = bottom_right[1];
            x = bottom_right[0];
            uc = Nu/2.0;
            vc = Nv/2.0;
            u = uc + focal_len*x/z;
            i2 = round(u)-1;
            v = vc + focal_len*y/z;
            j2 = round(v)-1;

    		// store parameters
    		string text;
    		text = "#Plane Parameters\n";
    		text += "A: " + boost::lexical_cast<std::string>(plane_parameters[0]) + "\n";
    		text += "B: " + boost::lexical_cast<std::string>(plane_parameters[1]) + "\n";
    		text += "C: " + boost::lexical_cast<std::string>(plane_parameters[2]) + "\n";
    		text += "D: " + boost::lexical_cast<std::string>(plane_parameters[3]) + "\n";
    		text += "cX: " + boost::lexical_cast<std::string>(plane_centroid[0]) + "\n";
    		text += "cY: " + boost::lexical_cast<std::string>(plane_centroid[1]) + "\n";
    		text += "cZ: " + boost::lexical_cast<std::string>(plane_centroid[2]) + "\n";
    		text += "#Box Filter Parameters\n";
    		text += "bnX: " + boost::lexical_cast<std::string>(boxn[0]) + "\n";
    		text += "bnY: " + boost::lexical_cast<std::string>(boxn[1]) + "\n";
    		text += "bnZ: " + boost::lexical_cast<std::string>(boxn[2]) + "\n";
    		text += "bmX: " + boost::lexical_cast<std::string>(boxm[0]) + "\n";
    		text += "bmY: " + boost::lexical_cast<std::string>(boxm[1]) + "\n";
    		text += "bmZ: " + boost::lexical_cast<std::string>(boxm[2]) + "\n";
    		text += "#Projection Parameters\n";
    		text += "Nv: " + boost::lexical_cast<std::string>(Nv) + "\n";
    		text += "Nu: " + boost::lexical_cast<std::string>(Nu) + "\n";
    		text += "focal_len: " + boost::lexical_cast<std::string>(focal_len) + "\n";
    		text += "mDis: " + boost::lexical_cast<std::string>(mDis) + "\n";
    		text += "nDis: " + boost::lexical_cast<std::string>(nDis) + "\n";
    		text += "#Cropped Image\n";
    		text += "tplf_x: " + boost::lexical_cast<std::string>(i) + "\n";
    		text += "tplf_y: " + boost::lexical_cast<std::string>(j) + "\n";
            text += "btrt_x: " + boost::lexical_cast<std::string>(i2) + "\n";
            text += "btrt_y: " + boost::lexical_cast<std::string>(j2) + "\n";
            text += "image_size: " + boost::lexical_cast<std::string>(image_size);
    		ofstream myfile;
    		myfile.open ("auto_recognition2.yaml");
    		myfile << text;
    		myfile.close();
            cout<<text<<endl;
            got_param = true;
			ROS_INFO("Select a new centroid!");
    	}
    	if(got_param)
    	{
    		// display
    		pcl::toROSMsg(*tf_kinect_ptr, ros_tf_kinect);
	    	//pcl::toROSMsg(*kinect_ptr, ros_kinect);
	    	pcl::toROSMsg(*tf_box_ptr, ros_tf_box);
	    	pcl::toROSMsg(*box_ptr, ros_box);

			//ros_kinect.header.frame_id = "camera";
			ros_tf_kinect.header.frame_id = "plane";
			ros_tf_box.header.frame_id = "plane";
			ros_box.header.frame_id = "camera";
			
			pub_kinect.publish(ros_kinect);
			pub_tf_kinect.publish(ros_tf_kinect);
			pub_tf_box.publish(ros_tf_box);
			pub_box.publish(ros_box);
			//ros::Duration(0.5).sleep();
    	}
    	pcl::toROSMsg(*kinect_ptr, ros_kinect);
    	ros_kinect.header.frame_id = "camera";
    	pub_kinect.publish(ros_kinect);
    	ros::spinOnce();
    	ros::Duration(1.0).sleep();
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
    ROS_INFO("The number of points within box = %d", n_extracted);    
    
}
