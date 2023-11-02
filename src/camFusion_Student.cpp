
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/transforms.h>
#include <pcl/search/kdtree.h>

#include "camFusion.hpp"
#include "dataStructures.h"
#include "processPointCloud.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints,
                        float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        // cout << "Number of Boxes: " << boundingBoxes.size() << endl;
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    cout << "3D objects boxes: " << boundingBoxes.size() << endl;
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // cout << "Looop 1" << endl;
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // cout << "Loop 2" << endl;
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }
        // cout << "End loops" << endl;

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }
    //cout << "End outer loop" << endl;

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    unsigned int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev,
                            std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    std::vector<cv::DMatch> matches;

    // is it in the bounding box
    vector<cv::DMatch> inBB;
    vector<double> distances;

    cv::KeyPoint curr;
    cv::KeyPoint prev;
    for (auto i = kptMatches.begin(); i != kptMatches.end(); ++i)
    {
        curr = kptsCurr[((*i)).trainIdx];
        prev = kptsPrev[((*i)).queryIdx];

        if (boundingBox.roi.contains(curr.pt))
        {
            inBB.push_back((*i));
            // compute distance
            distances.push_back(cv::norm(curr.pt - prev.pt));
        }
    }
    // compute mean
    double distTotal = std::accumulate(distances.begin(), distances.end(), 0.0);
    double mean = distTotal / distances.size();

    // filter out the ones with distance too much larger than mean
    double tolerance = 1.5 * mean;
    auto itrBB = inBB.begin();
    for (auto i = distances.begin(); i != distances.end(); ++i, ++itrBB)
    {
        if ((*i) < tolerance)
        {
            curr = kptsCurr[((*itrBB)).trainIdx];
            boundingBox.kptMatches.push_back((*itrBB));
            boundingBox.keypoints.push_back(curr);
        }
    }

}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    vector<double> distances;

    cv::KeyPoint curr, curr2;
    cv::KeyPoint prev, prev2;
    double distanceCurr, distancePrev;
    double min = 50.0;
    for (auto i = kptMatches.begin(); i != kptMatches.end()-1; ++i)
    {
        curr = kptsCurr[((*i)).trainIdx];
        prev = kptsPrev[((*i)).queryIdx];

        for (auto itr = kptMatches.begin() + 1; itr != kptMatches.end(); ++ itr)
        {
            curr2 = kptsCurr[((*itr)).trainIdx];
            prev2 = kptsPrev[((*itr)).queryIdx];

            distanceCurr = cv::norm(curr.pt - curr2.pt);
            distancePrev = cv::norm(prev.pt - prev2.pt);
        }

        if (distanceCurr >= min)
        {
            distances.push_back(distanceCurr/distancePrev);
        }
    }

    std::sort(distances.begin(), distances.end());
    int median = floor(distances.size() / 2);
    double medianRatio = distances.at(median);

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medianRatio);
}


void computeTTCLidarCluster(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // cluster points
    std::vector<std::vector<LidarPoint>> clusters = clustering(lidarPointsCurr, 0.3, 10, 1000);
    // cout << "Clustered" << endl;

    // find closest point
    typename pcl::PointCloud<pcl::PointXYZI>::Ptr points = pcl::PointCloud<pcl::PointXYZI>::Ptr();
    
    for (std::vector<LidarPoint> cluster : clusters)
    {
        for (LidarPoint point : cluster)
        {
            // if point in ego lane, lane width = 4 and in previous frame
            if ((std::abs(point.x) <= 2.0) && ((std::find(lidarPointsPrev.begin(), lidarPointsPrev.end(), point) != std::end(lidarPointsPrev))))
            {
                pcl::PointXYZI newPoint = pcl::PointXYZI({(float) point.x, (float) point.y, (float) point.z, (float) point.r});
                points->points.push_back(newPoint);
            }
        }
    }

    typename pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(points);
    // cout << "Made a tree" << endl;

    pcl::PointXYZI origin = pcl::PointXYZI({0.0, 0.0, 0.0, 1.0});
    vector<int> nearestIndices;
    vector<float> nearestDistances;
    tree->nearestKSearch(origin, 1, nearestIndices, nearestDistances);
    pcl::PointXYZI nearestPoint = points->points[nearestIndices.at(0)];

    //convert the point back to a lidar point
    LidarPoint nearestLidarPoint = LidarPoint({nearestPoint.x, nearestPoint.y, nearestPoint.z, nearestPoint.intensity});
    // find its index in lidarPointsCurr
    auto it = std::find(lidarPointsCurr.begin(), lidarPointsCurr.end(), nearestLidarPoint);
    int index;
    if (it != lidarPointsCurr.end())
    {
        index = std::distance(lidarPointsCurr.begin(), it);
    }
    // TTC = x * (1/frameRate)/(xPrev - xCurr)
    TTC = lidarPointsCurr.at(index).x * ((1/frameRate) / (lidarPointsPrev.at(index).x - lidarPointsCurr.at(index).x));
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double dT = 1 / frameRate;
    double laneWidth = 4.0;

    std::vector<double> inLaneCurr;
    std::vector<double> inLanePrev;

    // if in lane, keep it
    for (LidarPoint point : lidarPointsCurr)
    {
        if (std::abs(point.y/2) <= 2)
        {
            inLaneCurr.push_back(point.x);
        }
    }
    // std::cout << "Tested inLane Curr" << std::endl;
    for (LidarPoint point : lidarPointsPrev)
    {
        if (std::abs(point.y/2) <= 2)
        {
            inLanePrev.push_back(point.x);
        }
    }
    // std::cout << "Tested inLanePrev" << std::endl;

    // sort by closest y (closest to ego car)
    sort(inLaneCurr.begin(), inLaneCurr.end());
    sort(inLanePrev.begin(), inLanePrev.end());

    // std::cout << "Sorted" << std::endl;

    // select median-ish index
    int medianIndex = floor(inLanePrev.size()/2.0);

    //std::cout << "Incoming prev size: " << lidarPointsPrev.size() << std::endl;
    // std::cout << "In Lane Size: " << inLanePrev.size() << std::endl;
    // std::cout << "Median: " << medianIndex << std::endl;

    double d1 = inLaneCurr.at(medianIndex);
    double d0 = inLanePrev.at(medianIndex);

    TTC = d1 * dT / (d0-d1);

}
void matchBoundingBoxes(std::vector<cv::DMatch> &matches,
    std::map<int, int> &bbBestMatches, DataFrame &prevFrame,
    DataFrame &currFrame)
{
    std::multimap<int, int> multimap;
    
    std::vector<cv::KeyPoint> currentKeypoints = currFrame.keypoints;
    std::vector<cv::KeyPoint> prevKeypoints = prevFrame.keypoints;

    std::vector<BoundingBox> currentBBoxList = currFrame.boundingBoxes;
    std::vector<BoundingBox> prevBBoxList = prevFrame.boundingBoxes;

    // outer loop through matches
    for (cv::DMatch match : matches)
    {
        int currentId = match.trainIdx;
        int prevId = match.queryIdx;

        cv::Point2f currentPoint = currentKeypoints.at(currentId).pt;
        cv::Point2f prevPoint = prevKeypoints.at(prevId).pt;

        int prevBoxID = -1;
        int currBoxID = -1;

        // find out by which bounding box each kpt is enclosed in prev
        // and current frame
        for (BoundingBox prevBox : prevBBoxList)
        {
            if (prevBox.roi.contains(prevPoint))
            {
                prevBoxID = prevBox.boxID;
            }   
        }
        for (BoundingBox currBox : currentBBoxList)
        {
            if (currBox.roi.contains(currentPoint))
            {
                currBoxID = currBox.boxID;
            }
        }
        if (prevBoxID != -1 && currBoxID != -1)
        {
            multimap.insert(std::make_pair(currBoxID, prevBoxID));
        }
    }

    for (BoundingBox currBox : currentBBoxList)
    {
        auto matched_idx = multimap.equal_range(currBox.boxID);
        std::vector<int> cnt(prevBBoxList.size() + 1, 0);

        for (auto itr = matched_idx.first; itr != matched_idx.second; itr++)
        {
            cnt[(*itr).second] += 1;
        }

        bbBestMatches.insert(std::make_pair(std::distance(cnt.begin(), std::max_element(cnt.begin(), cnt.end())), currBox.boxID));
    }
}
