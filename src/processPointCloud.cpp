// point cloud processing - parts taken from assignment in earlier lidar course project

#include "processPointCloud.h"

std::vector<std::vector<LidarPoint>> clustering(std::vector<LidarPoint> cloud, float clusterTolerance, int minSize, int maxSize)
{
    // turn LidarPoints into pcl::PointXYZI and add to 
    typename pcl::PointCloud<pcl::PointXYZI>::Ptr points = pcl::PointCloud<pcl::PointXYZI>::Ptr();
    
    for (LidarPoint point : cloud)
    {
        pcl::PointXYZI newPoint = pcl::PointXYZI({(float) point.x, (float) point.y, (float) point.z, (float) point.r});
        points->points.push_back(newPoint);
    }

    std::vector<typename pcl::PointCloud<pcl::PointXYZI>> clusters;

    typename pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(points);

    std::vector<pcl::PointIndices> clusterIndices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(clusterTolerance);
    ec.setMinClusterSize(minSize);
    ec.setMaxClusterSize(maxSize);
    ec.setSearchMethod(tree);
    ec.setInputCloud(points);
    ec.extract(clusterIndices);

    for(pcl::PointIndices getIndices: clusterIndices)
    {
        typename pcl::PointCloud<pcl::PointXYZI> cloudCluster = pcl::PointCloud<pcl::PointXYZI>();

        for (int index : getIndices.indices)
            cloudCluster.points.push_back(points->points[index]);

        cloudCluster.width = cloudCluster.points.size();
        cloudCluster.height = 1;
        cloudCluster.is_dense = true;

        clusters.push_back(cloudCluster);
    }

    // change clusters and points back into vectors of LidarPoints
    std::vector<std::vector<LidarPoint>> lidarClusters;
    for (typename pcl::PointCloud<pcl::PointXYZI> cluster : clusters)
    {
        std::vector<LidarPoint> lidarCluster;
        for (pcl::PointXYZI point : cluster)
        {
            LidarPoint lidarPoint = LidarPoint{(double) point.x, (double) point.y, (double) point.z, (double) point.intensity};
            lidarCluster.push_back(lidarPoint);
        }
        lidarClusters.push_back(lidarCluster);
    }

    return lidarClusters;
}