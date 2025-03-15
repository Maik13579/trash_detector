#include "trash_detector_node.hpp"
#include <pcl/common/common.h>
#include <pcl/filters/crop_box.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <Eigen/Dense>
#include <tf2_eigen/tf2_eigen.h>

TrashDetector::TrashDetector(ros::NodeHandle &nh)
    : nh_(nh), tf_listener_(tf_buffer_)
{
    loadParameters();
    cloud_sub_ = nh_.subscribe("/xtion/depth_registered/points", 1, &TrashDetector::cloudCallback, this);
    service_srv_ = nh_.advertiseService("detect_trash", &TrashDetector::serviceCallback, this);
    marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("trash_markers", 1);
}

void TrashDetector::loadParameters()
{
    ros::NodeHandle pnh("~");
    pnh.param<std::string>("base_frame", base_frame_, "base_link");

    pnh.param("bounding_box/min_x", bb_min_x_, -5.0);
    pnh.param("bounding_box/max_x", bb_max_x_, 5.0);
    pnh.param("bounding_box/min_y", bb_min_y_, -5.0);
    pnh.param("bounding_box/max_y", bb_max_y_, 5.0);
    pnh.param("bounding_box/min_z", bb_min_z_, -1.0);
    pnh.param("bounding_box/max_z", bb_max_z_, 2.0);

    pnh.param("ransac/use_ransac", use_ransac_, true);
    pnh.param("ransac/distance_threshold", ransac_distance_threshold_, 0.02);
    pnh.param("ransac/max_iterations", ransac_max_iterations_, 100);

    pnh.param("clustering/cluster_tolerance", cluster_tolerance_, 0.02);
    pnh.param("clustering/min_cluster_size", min_cluster_size_, 50);
    pnh.param("clustering/max_cluster_size", max_cluster_size_, 10000);

    pnh.param("final_filter/min_side_length", min_side_length_, 0.1);
    pnh.param("final_filter/max_side_length", max_side_length_, 2.0);
    pnh.param("final_filter/min_height", min_height_, 0.1);
    pnh.param("final_filter/max_height", max_height_, 2.0);
}

void TrashDetector::cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
    latest_cloud_ = cloud_msg;
}


bool TrashDetector::serviceCallback(trash_detector_interfaces::DetectTrash::Request& req,
    trash_detector_interfaces::DetectTrash::Response& res)
{
    if (!latest_cloud_) {
        ROS_WARN("No point cloud received yet.");
        return false;
    }

    // Lookup transform from the cloud frame to base_frame
    geometry_msgs::TransformStamped transformStamped;
    try {
        transformStamped = tf_buffer_.lookupTransform(base_frame_, latest_cloud_->header.frame_id,
                            ros::Time(0), ros::Duration(1.0));
    } catch (tf2::TransformException &ex) {
        ROS_ERROR("lookupTransform failed: %s", ex.what());
        return false;
    }

    // Convert the incoming cloud to a PCL point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*latest_cloud_, *pcl_cloud);

    // Convert geometry_msgs::Transform to Eigen::Isometry3d manually
    Eigen::Isometry3d eigen_transform = Eigen::Isometry3d::Identity();
    eigen_transform.translate(Eigen::Vector3d(
        transformStamped.transform.translation.x,
        transformStamped.transform.translation.y,
        transformStamped.transform.translation.z));
    Eigen::Quaterniond q(
        transformStamped.transform.rotation.w,
        transformStamped.transform.rotation.x,
        transformStamped.transform.rotation.y,
        transformStamped.transform.rotation.z);
    eigen_transform.rotate(q);
    Eigen::Matrix4f transform = eigen_transform.matrix().cast<float>();

    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*pcl_cloud, *pcl_cloud_transformed, transform);

    // Crop the cloud using a bounding box
    pcl::CropBox<pcl::PointXYZ> crop;
    crop.setMin(Eigen::Vector4f(bb_min_x_, bb_min_y_, bb_min_z_, 1.0));
    crop.setMax(Eigen::Vector4f(bb_max_x_, bb_max_y_, bb_max_z_, 1.0));
    crop.setInputCloud(pcl_cloud_transformed);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    crop.filter(*cropped_cloud);

    // Optionally remove floor via RANSAC
    pcl::PointCloud<pcl::PointXYZ>::Ptr no_floor_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    if (use_ransac_) {
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(ransac_distance_threshold_);
        seg.setMaxIterations(ransac_max_iterations_);
        seg.setInputCloud(cropped_cloud);
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.empty()) {
            ROS_WARN("No plane found using RANSAC.");
            no_floor_cloud = cropped_cloud;
        } else {
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud(cropped_cloud);
            extract.setIndices(inliers);
            extract.setNegative(true);
            extract.filter(*no_floor_cloud);
        }
    } else {
        no_floor_cloud = cropped_cloud;
    }

    // Cluster extraction using Euclidean clustering
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(no_floor_cloud);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(no_floor_cloud);
    ec.extract(cluster_indices);

    // Create MarkerArray to hold bounding boxes for valid clusters
    visualization_msgs::MarkerArray marker_array;
    int id = 0;
    ros::Time stamp = ros::Time::now();
    for (const auto& indices : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>());
        for (int index : indices.indices)
            cluster->points.push_back(no_floor_cloud->points[index]);
        if (cluster->points.empty())
            continue;

        // Compute min and max z for height
        float min_z_val = std::numeric_limits<float>::max();
        float max_z_val = -std::numeric_limits<float>::max();
        for (const auto& pt : cluster->points) {
            if (pt.z < min_z_val) min_z_val = pt.z;
            if (pt.z > max_z_val) max_z_val = pt.z;
        }
        float height_box = max_z_val - min_z_val;

        // Because we are in base_frame we can assume that z is looking upwards
        // So we compute the minimal oriented bounding box just using x and y

        // Prepare matrix for XY points
        Eigen::MatrixXf pts(2, cluster->points.size());
        for (size_t i = 0; i < cluster->points.size(); i++) {
            pts(0, i) = cluster->points[i].x;
            pts(1, i) = cluster->points[i].y;
        }
        // Compute mean and center the data
        Eigen::Vector2f mean = pts.rowwise().mean();
        Eigen::MatrixXf centered = pts.colwise() - mean;
        // Compute covariance matrix and eigen decomposition
        Eigen::Matrix2f cov = centered * centered.transpose() / static_cast<float>(cluster->points.size());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(cov);
        Eigen::Matrix2f eig_vec = es.eigenvectors();
        // Largest eigenvector (for maximum variance) is in column 1 (eigenvalues sorted in increasing order)
        Eigen::Vector2f axis1 = eig_vec.col(1);
        Eigen::Vector2f axis2 = eig_vec.col(0);
        // Form rotation matrix (columns are the eigenvectors)
        Eigen::Matrix2f R;
        R << axis1(0), axis2(0),
             axis1(1), axis2(1);
        // Rotate points into the principal axes frame
        Eigen::MatrixXf rotated = R.transpose() * centered;
        float min_x_val = rotated.row(0).minCoeff();
        float max_x_val = rotated.row(0).maxCoeff();
        float min_y_val = rotated.row(1).minCoeff();
        float max_y_val = rotated.row(1).maxCoeff();
        float width_box = max_x_val - min_x_val;
        float depth_box = max_y_val - min_y_val;
        // Compute center in rotated coordinates and transform back
        float center_x_rot = (min_x_val + max_x_val) / 2.0f;
        float center_y_rot = (min_y_val + max_y_val) / 2.0f;
        Eigen::Vector2f center_rot(center_x_rot, center_y_rot);
        Eigen::Vector2f center_xy = R * center_rot + mean;

        // Filter clusters based on bounding box dimensions
        if (width_box < min_side_length_ || width_box > max_side_length_ ||
            depth_box < min_side_length_ || depth_box > max_side_length_ ||
            height_box < min_height_ || height_box > max_height_)
            continue;

        // Create marker for the bounding box
        visualization_msgs::Marker marker;
        marker.header.frame_id = base_frame_;
        marker.header.stamp = stamp;
        marker.ns = "trash";
        marker.id = id++;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        // Set position: center XY from PCA and z center from min and max z.
        marker.pose.position.x = center_xy(0);
        marker.pose.position.y = center_xy(1);
        marker.pose.position.z = (min_z_val + max_z_val) / 2.0f;
        // Set orientation: rotation about z using the angle of axis1.
        double theta = std::atan2(axis1(1), axis1(0));
        tf2::Quaternion q;
        q.setRPY(0, 0, theta);
        marker.pose.orientation.x = q.x();
        marker.pose.orientation.y = q.y();
        marker.pose.orientation.z = q.z();
        marker.pose.orientation.w = q.w();
        marker.scale.x = width_box;
        marker.scale.y = depth_box;
        marker.scale.z = height_box;
        marker.color.a = 0.5;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;

        marker_array.markers.push_back(marker);
    }

    marker_pub_.publish(marker_array);
    res.markers = marker_array;
    return true;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "trash_detector_node");
    ros::NodeHandle nh;
    TrashDetector detector(nh);
    ros::spin();
    return 0;
}
