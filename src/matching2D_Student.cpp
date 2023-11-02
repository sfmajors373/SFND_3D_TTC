
#include <numeric>
#include "matching2D.hpp"

using namespace std;

void bVisFunction(string windowName, cv::Mat &img, std::vector<cv::KeyPoint> &keypoints)
{
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
}


// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    cv::Mat descSourceFloat = descSource.clone();
    cv::Mat descRefFloat = descRef.clone();

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        {
            descSourceFloat.convertTo(descSourceFloat, CV_32F);
            descRefFloat.convertTo(descRefFloat, CV_32F);
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2);
        
        // descriptor distance ratio filtering 
        for (int i = 0; i < knn_matches.size(); i++)
        {
            float dist1 = knn_matches[i][0].distance;
            float dist2 = knn_matches[i][1].distance;

            if ((dist1 / dist2) < 0.8)
            {
                matches.push_back(knn_matches[i][0]);
            }
        }

        std::cout << "Number matches: " << matches.size() << std::endl;
    }
}


//// TODO: ORB, FREAK, AKAZE, SIFT
// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }
    else
    {
        return;
    }
    

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        bVisFunction("Shi-Tomasi detector", img, keypoints);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    int blockSize = 2;
    int aperatureSize = 3;
    int minResponse = 100;
    double k = 0.04;
    cv::Mat dst = cv::Mat::zeros(img.size(), CV_32FC1);

    double t = (double)cv::getTickCount();
    cv::cornerHarris(img, dst, blockSize, aperatureSize, k, cv::BORDER_DEFAULT);

    cv::Mat dst_norm, dst_norm_scaled;
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    float maxOverlap = 0.0;

    for (int r = 0; r < dst_norm.rows; r++)
    {
        for (int c = 0; c < dst_norm.cols; c++)
        {
            int response = (int)dst_norm.at<float>(r,c);
            if (response > minResponse)
            {
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(c,r);
                newKeyPoint.size = 2 * aperatureSize;
                newKeyPoint.response = response;

                // non-max supression
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    float kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {
                            *it = newKeyPoint;
                            break;
                        }
                    }
                }
                if (!bOverlap)
                {
                    keypoints.push_back(newKeyPoint);
                }
            }

        }   
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis)
    {
        string windowName = "Harris Corner Detection Results";
        cv::namedWindow(windowName, 5);
        cv::Mat visImage = dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsBrisk(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // need to use the pointer because of weirdness in the implementation of BRISK in openCV
    double t = (double)cv::getTickCount();
    cv::Ptr<cv::BRISK> BriskDetector = cv::BRISK::create();
    BriskDetector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Brisk detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis)
    {
        bVisFunction("Brisk Keypoint Detector", img, keypoints);
    }
}

void detKeypointsFast(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    double t = (double)cv::getTickCount();
    cv::Ptr<cv::FastFeatureDetector> FastDetector = cv::FastFeatureDetector::create();
    FastDetector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Fast detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    if (bVis)
    {
        bVisFunction("Fast Keypoint Detector", img, keypoints);
    }
}

void detKeypointsOrb(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    double t = (double)cv::getTickCount();
    cv::Ptr<cv::ORB> OrbDetector = cv::ORB::create();
    OrbDetector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Orb detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis)
    {
        bVisFunction("ORB Keypoint Detector", img, keypoints);
    }
}

void detKeypointsAkaze(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    double t = (double)cv::getTickCount();
    cv::Ptr<cv::AKAZE> AkazeDetector = cv::AKAZE::create();
    AkazeDetector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Akaze detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis)
    {
        bVisFunction("AKAZE Keypoint Detector", img, keypoints);
    }
}

void detKeypointsSift(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    double t = (double)cv::getTickCount();
    cv::Ptr<cv::xfeatures2d::SIFT> SiftDetector = cv::xfeatures2d::SIFT::create();
    SiftDetector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Sift detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis)
    {
        bVisFunction("SIFT Keypoint Detector", img, keypoints);
    }
}