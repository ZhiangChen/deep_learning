cmake_minimum_required(VERSION 2.8.3)
project(data_pp)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  pcl_ros
  pcl_conversions
  pcl_msgs
  sensor_msgs
  std_msgs
  tf
  cv_bridge
  image_transport
  pcl_utils
)
find_package(PCL 1.7 REQUIRED)
find_package(cmake_modules REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Eigen REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
add_definitions(${EIGEN_DEFINITIONS})
include_directories(${PCL_INCLUDE_DIRS})

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(data_pp src/data_pp.cpp)

 target_link_libraries(data_pp
   ${catkin_LIBRARIES}
 )
