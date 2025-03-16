#ifndef TRASH_DETECTOR_NODE_HPP
#define TRASH_DETECTOR_NODE_HPP

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <trash_detector_interfaces/DetectTrash.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <pcl_ros/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>

class TrashDetector
{
public:
    TrashDetector(ros::NodeHandle& nh);
    bool serviceCallback(trash_detector_interfaces::DetectTrash::Request& req,
                         trash_detector_interfaces::DetectTrash::Response& res);
    bool trashCanServiceCallback(trash_detector_interfaces::DetectTrash::Request& req,
                                 trash_detector_interfaces::DetectTrash::Response& res);
private:
    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg);
    void loadParameters();

    ros::NodeHandle nh_;
    ros::Subscriber cloud_sub_;
    ros::ServiceServer service_srv_;
    ros::ServiceServer trash_can_srv_;
    ros::Publisher marker_pub_;
    ros::Publisher trash_can_pub_;

    sensor_msgs::PointCloud2::ConstPtr latest_cloud_;

    // Parameters
    std::string base_frame_;
    // Bounding box parameters
    double bb_min_x_, bb_max_x_, bb_min_y_, bb_max_y_, bb_min_z_, bb_max_z_;

    // RANSAC parameters
    bool use_ransac_;
    double ransac_distance_threshold_;
    int ransac_max_iterations_;

    // Clustering parameters
    double cluster_tolerance_;
    int min_cluster_size_;
    int max_cluster_size_;

    // Final filter parameters
    double min_side_length_, max_side_length_;
    double min_height_, max_height_;

    // trash can dimensions 
    double trash_can_height_;
    double trash_can_bottom_radius_;
    double trash_can_top_radius_;
    int num_layers_;        // Layers along height.
    int points_per_layer_;

    // ICP parameters for trash can detection
    int icp_max_iterations_;
    double icp_transformation_epsilon_;
    std::vector<double> icp_corr_distances_;
    double icp_threshold_;

    // TF2 buffer and listener for transformations
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
};

#endif // TRASH_DETECTOR_NODE_HPP
