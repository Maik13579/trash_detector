#include "trash_detector_node.hpp"
#include <pcl/common/common.h>
#include <pcl/filters/crop_box.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
#include <tf2_eigen/tf2_eigen.h>
#include <cstdlib>
#include <cmath>
#include <algorithm>

// Helper function to compute the inlier ratio (percent of source points with a correspondence)
// using a given threshold (in meters). Distances are squared in the kdtree search.
double computeInlierRatio(const pcl::PointCloud<pcl::PointXYZ>::Ptr &source,
                          const pcl::PointCloud<pcl::PointXYZ>::Ptr &target,
                          double threshold)
{
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(target);
    int inlierCount = 0;
    double threshold_sqr = threshold * threshold;
    for (const auto &pt : source->points)
    {
        std::vector<int> idx;
        std::vector<float> dists;
        if (kdtree.nearestKSearch(pt, 1, idx, dists) > 0)
        {
            // dists are squared distances.
            if (dists[0] < threshold_sqr)
                inlierCount++;
        }
    }
    return static_cast<double>(inlierCount) / static_cast<double>(source->points.size());
}

TrashDetector::TrashDetector(ros::NodeHandle &nh)
    : nh_(nh), tf_listener_(tf_buffer_)
{
    loadParameters();
    cloud_sub_ = nh_.subscribe("/xtion/depth_registered/points", 1, &TrashDetector::cloudCallback, this);
    service_srv_ = nh_.advertiseService("detect_trash", &TrashDetector::serviceCallback, this);
    trash_can_srv_ = nh_.advertiseService("detect_trash_can", &TrashDetector::trashCanServiceCallback, this);
    marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("trash_markers", 1);
    trash_can_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("trash_can", 1);
}

void TrashDetector::loadParameters()
{
    ros::NodeHandle pnh("~");
    pnh.param<std::string>("base_frame", base_frame_, "base_link");

    // Base frame for transformations.
    pnh.param<std::string>("base_frame", base_frame_, "base_link");

    // Bounding box parameters to crop the input point cloud.
    pnh.param("bounding_box/min_x", bb_min_x_, -5.0);
    pnh.param("bounding_box/max_x", bb_max_x_, 5.0);
    pnh.param("bounding_box/min_y", bb_min_y_, -5.0);
    pnh.param("bounding_box/max_y", bb_max_y_, 5.0);
    pnh.param("bounding_box/min_z", bb_min_z_, -1.0);
    pnh.param("bounding_box/max_z", bb_max_z_, 2.0);

    // RANSAC parameters for floor removal.
    pnh.param("ransac/use_ransac", use_ransac_, true);
    pnh.param("ransac/distance_threshold", ransac_distance_threshold_, 0.02);
    pnh.param("ransac/max_iterations", ransac_max_iterations_, 100);

    // Euclidean clustering parameters.
    pnh.param("clustering/cluster_tolerance", cluster_tolerance_, 0.02);
    pnh.param("clustering/min_cluster_size", min_cluster_size_, 50);
    pnh.param("clustering/max_cluster_size", max_cluster_size_, 10000);

    // Trash filter dimensions for valid objects.
    pnh.param("trash_filter/min_side_length", min_side_length_, 0.1);
    pnh.param("trash_filter/max_side_length", max_side_length_, 2.0);
    pnh.param("trash_filter/min_height", min_height_, 0.1);
    pnh.param("trash_filter/max_height", max_height_, 2.0);

    // Trash can detection parameters.
    pnh.param("trash_can/height", trash_can_height_, 1.0);
    pnh.param("trash_can/bottom_radius", trash_can_bottom_radius_, 0.2);
    pnh.param("trash_can/top_radius", trash_can_top_radius_, 0.3);
    pnh.param("trash_can/num_layers", num_layers_, 20);             // Layers along height.
    pnh.param("trash_can/points_per_layer", points_per_layer_, 36);   // Points per layer.

    // ICP parameters for trash can detection.
    pnh.param("icp/max_iterations", icp_max_iterations_, 50);
    pnh.param("icp/transformation_epsilon", icp_transformation_epsilon_, 1e-8);
    std::vector<double> icp_corr_distances;
    if (!pnh.getParam("icp/corr_distances", icp_corr_distances)) {
        // Default values if the parameter isn't set.
        icp_corr_distances = {0.3, 0.1, 0.05, 0.02};
    }
    icp_corr_distances_ = icp_corr_distances;
    pnh.param("icp/min_threshold", icp_threshold_, 0.1);
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

bool TrashDetector::trashCanServiceCallback(trash_detector_interfaces::DetectTrash::Request& req,
                                              trash_detector_interfaces::DetectTrash::Response& res)
{
    if (!latest_cloud_) {
        ROS_WARN("No point cloud received yet.");
        return false;
    }

    // Lookup transform from cloud frame to base_frame.
    geometry_msgs::TransformStamped transformStamped;
    try {
        transformStamped = tf_buffer_.lookupTransform(base_frame_, latest_cloud_->header.frame_id,
                                                      ros::Time(0), ros::Duration(1.0));
    } catch (tf2::TransformException &ex) {
        ROS_ERROR("lookupTransform failed: %s", ex.what());
        return false;
    }

    // Convert incoming cloud to PCL.
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*latest_cloud_, *pcl_cloud);

    // Convert geometry_msgs::Transform to Eigen::Isometry3d manually.
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

    // Crop the cloud using a bounding box.
    pcl::CropBox<pcl::PointXYZ> crop;
    crop.setMin(Eigen::Vector4f(bb_min_x_, bb_min_y_, bb_min_z_, 1.0));
    crop.setMax(Eigen::Vector4f(bb_max_x_, bb_max_y_, bb_max_z_, 1.0));
    crop.setInputCloud(pcl_cloud_transformed);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    crop.filter(*cropped_cloud);

    // Optionally remove floor via RANSAC.
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
            ROS_WARN("No floor plane found.");
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

    // Cluster extraction using Euclidean clustering.
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

    // Generate synthetic model point cloud for the expected trash can.
    pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    int num_z = 20;   // number of layers along height
    int num_theta = 36; // points per layer
    for (int i = 0; i < num_z; i++) {
        double t = static_cast<double>(i) / (num_z - 1);
        double z = t * trash_can_height_;
        double radius = trash_can_bottom_radius_ + t * (trash_can_top_radius_ - trash_can_bottom_radius_);
        for (int j = 0; j < num_theta; j++) {
            double theta = 2 * M_PI * j / num_theta;
            pcl::PointXYZ pt;
            pt.x = radius * cos(theta);
            pt.y = radius * sin(theta);
            pt.z = z;
            model_cloud->points.push_back(pt);
        }
    }

    // Initialize variables to store the best alignment result.
    double best_ratio = -1.0;
    Eigen::Matrix4f best_transform = Eigen::Matrix4f::Identity();
    pcl::PointCloud<pcl::PointXYZ>::Ptr best_cluster(new pcl::PointCloud<pcl::PointXYZ>());
    
    // Use the list of ICP correspondence distances.
    std::vector<double> icp_corr_distances = icp_corr_distances_;

    // Iterate over each cluster and perform multi-scale ICP alignment.
    for (const auto& indices : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::copyPointCloud(*no_floor_cloud, indices, *cluster);
        if (cluster->empty())
            continue;
    
        // Align the cluster's centroid with the trash can base (z = 0).
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cluster, centroid);
        centroid[2] = 0.0;
        Eigen::Matrix4f transform_icp = Eigen::Matrix4f::Identity();
        transform_icp.block<3,1>(0,3) = centroid.head<3>();
    
        pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::transformPointCloud(*model_cloud, *model_cloud_transformed, transform_icp);
    
        // Multi-scale ICP loop using inlier ratio as metric.
        Eigen::Matrix4f accumulated_transform = Eigen::Matrix4f::Identity();
        double current_ratio = 0.0;
        bool converged = true;
    
        for (double corr_dist : icp_corr_distances) {
            pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
            icp.setInputSource(model_cloud_transformed);
            icp.setInputTarget(cluster);
            icp.setMaximumIterations(icp_max_iterations_);
            icp.setTransformationEpsilon(icp_transformation_epsilon_);
            icp.setMaxCorrespondenceDistance(corr_dist);
    
            pcl::PointCloud<pcl::PointXYZ> aligned;
            icp.align(aligned);
    
            if (!icp.hasConverged()) {
                converged = false;
                break;
            }
    
            // Update the accumulated transform.
            accumulated_transform = icp.getFinalTransformation() * accumulated_transform;
            pcl::transformPointCloud(*model_cloud_transformed, *model_cloud_transformed, icp.getFinalTransformation());
    
            // Compute inlier ratio: percentage of source points (transformed model) with a correspondence in the cluster.
            current_ratio = computeInlierRatio(model_cloud_transformed, cluster, corr_dist);
        }
    
        // Keep the best alignment (highest inlier ratio).
        if (converged && current_ratio > best_ratio) {
            best_ratio = current_ratio;
            best_cluster = cluster;
            best_transform = accumulated_transform * transform_icp;
        }
    }

    if (best_ratio < icp_threshold_) {
        ROS_INFO("No trash can detected.");
        return false;
    }
    
    // Force trash can upright (remove tilt) and place on the floor.
    Eigen::Vector3f translation = best_transform.block<3,1>(0,3);
    Eigen::Matrix3f rotation = best_transform.block<3,3>(0,0);
    float yaw = rotation.eulerAngles(0, 1, 2)[2];
    
    Eigen::Affine3f upright_transform = Eigen::Affine3f::Identity();
    upright_transform.translation() = translation;
    upright_transform.linear() = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()).toRotationMatrix();
    
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(*model_cloud, min_pt, max_pt);
    upright_transform.translation().z() = -min_pt.z;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr best_aligned_model(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*model_cloud, *best_aligned_model, upright_transform);
    
    sensor_msgs::PointCloud2 aligned_msg;
    pcl::toROSMsg(*best_aligned_model, aligned_msg);
    aligned_msg.header.frame_id = base_frame_;
    aligned_msg.header.stamp = ros::Time::now();
    trash_can_pub_.publish(aligned_msg);
    
    // Create a cylinder marker for visualizing the trash can.
    visualization_msgs::MarkerArray marker_array;
    ros::Time stamp = ros::Time::now();
    
    if (!best_cluster->points.empty()) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = base_frame_;
        marker.header.stamp = stamp;
        marker.ns = "trash_can";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::CYLINDER;
        marker.action = visualization_msgs::Marker::ADD;
    
        marker.pose.position.x = upright_transform.translation().x();
        marker.pose.position.y = upright_transform.translation().y();
        marker.pose.position.z = trash_can_height_ / 2.0; // Grounded on the floor.
    
        Eigen::Quaternionf quat(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
        marker.pose.orientation.x = quat.x();
        marker.pose.orientation.y = quat.y();
        marker.pose.orientation.z = quat.z();
        marker.pose.orientation.w = quat.w();
    
        marker.scale.z = trash_can_height_;
        marker.scale.x = marker.scale.y = (trash_can_bottom_radius_ + trash_can_top_radius_);
    
        // Color the marker based on the best inlier ratio.
        marker.color.r = 1.0 - best_ratio;
        marker.color.g = best_ratio;
        marker.color.b = 0.0;
        marker.color.a = 0.8;
    
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
