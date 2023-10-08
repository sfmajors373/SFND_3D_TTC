// PCL lib functions for processing point clouds: parts and ideas taken from lidar course earlier in nano-degree

#ifndef PROCESSPOINTCLOUDS_H_
#define PROCESSPOINTCLOUDS_H_

#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/transforms.h>
#include "dataStructures.h"

std::vector<std::vector<LidarPoint>> clustering(std::vector<LidarPoint> lidarPoints, float clusterTolerance, int minSize, int maxSize);

#endif