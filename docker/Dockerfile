FROM ros:noetic

ENV DEBIAN_FRONTEND=noninteractive

# Install build tools and necessary packages including missing ROS dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-rosdep \
    python3-catkin-tools \
    git \
    ros-noetic-tf2-ros \
    ros-noetic-pcl-ros \
    ros-noetic-pcl-conversions \
    && rm -rf /var/lib/apt/lists/*

RUN rm -f /etc/ros/rosdep/sources.list.d/20-default.list && \
    rosdep init && rosdep update

# Create a catkin workspace
WORKDIR /root/catkin_ws
RUN mkdir -p src

COPY trash_detector src/trash_detector
COPY trash_detector_interfaces src/trash_detector_interfaces

# Install dependencies using rosdep 
RUN rosdep install --from-paths src --ignore-src -y -r

# Build the workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"
CMD ["/bin/bash", "-c", "source /root/catkin_ws/devel/setup.bash && roslaunch trash_detector trash_detector.launch"]