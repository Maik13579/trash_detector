cmake_minimum_required(VERSION 3.0.2)
project(trash_detector)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  visualization_msgs
  tf2_ros
  pcl_ros
  pcl_conversions
  trash_detector_interfaces
)

find_package(PCL REQUIRED)
find_package(tf2_eigen REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs visualization_msgs tf2_ros pcl_ros pcl_conversions trash_detector_interfaces
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${tf2_eigen_INCLUDE_DIRS}
)

add_executable(trash_detector_node src/trash_detector_node.cpp)
add_dependencies(trash_detector_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(trash_detector_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${tf2_eigen_LIBRARIES}
)
