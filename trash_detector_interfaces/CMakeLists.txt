cmake_minimum_required(VERSION 3.0.2)
project(trash_detector_interfaces)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  visualization_msgs
)

add_service_files(
  FILES
  DetectTrash.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  visualization_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs visualization_msgs
)
