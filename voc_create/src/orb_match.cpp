#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <vector>
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
bool refineMatchesWithHomography(
    const std::vector<cv::KeyPoint> &queryKeypoints,
    const std::vector<cv::KeyPoint> &trainKeypoints,
    float reprojectionThreshold, std::vector<cv::DMatch> &matches,
    cv::Mat &homography)
{
    const int minNumberMatchesAllowed = 8;

    if (matches.size() < minNumberMatchesAllowed)
        return false;

    // Prepare data for cv::findHomography
    std::vector<cv::Point2f> srcPoints(matches.size());
    std::vector<cv::Point2f> dstPoints(matches.size());

    for (size_t i = 0; i < matches.size(); i++)
    {
        srcPoints[i] = trainKeypoints[matches[i].trainIdx].pt;
        dstPoints[i] = queryKeypoints[matches[i].queryIdx].pt;
    }

    // Find homography matrix and get inliers mask
    std::vector<unsigned char> inliersMask(srcPoints.size());
    homography = cv::findHomography(srcPoints, dstPoints, CV_FM_RANSAC,
                                    reprojectionThreshold, inliersMask);

    std::vector<cv::DMatch> inliers;
    for (size_t i = 0; i < inliersMask.size(); i++)
    {
        if (inliersMask[i])
            inliers.push_back(matches[i]);
    }

    matches.swap(inliers);
    return matches.size() > minNumberMatchesAllowed;
}

void draw_good_match(cv::Mat &img1,const vector<KeyPoint> &kp1,cv::Mat &des1,cv::Mat &img2,const vector<KeyPoint> &kp2,cv::Mat &des2)
{
 vector<DMatch> matches;
    BFMatcher bfMatcher(NORM_HAMMING);        
    bfMatcher.match(des1, des2, matches);

    Mat matHomo;
    refineMatchesWithHomography(kp1, kp2, 3, matches, matHomo);
    cout << "[Info] Homography T : " << matHomo << endl;
    cout << "[Info] # of matches : " << matches.size() << endl;

    Mat imResult;
    drawMatches(img1, kp1, img2, kp2, matches, imResult);

    imshow("refined matches", imResult);
    imwrite("refined_orb_matches_92_95.jpg", imResult);
    waitKey();
    
}

int main( int argc, char* argv[] )
{
    Mat img1 = imread( "/home/gxf/slam/BASD_voc_create/voc_create/src/000092.png", IMREAD_GRAYSCALE );
    Mat img2 = imread("/home/gxf/slam/BASD_voc_create/voc_create/src/000098.png", IMREAD_GRAYSCALE );
    if ( img1.empty() || img2.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        return -1;
    }
    Ptr<ORB> detector = ORB::create(1000);
    std::vector<KeyPoint> kp1, kp2;
    Mat des1, des2;
    detector->detectAndCompute( img1, noArray(), kp1, des1 );
    detector->detectAndCompute( img2, noArray(), kp2, des2 );
    draw_good_match(img1,kp1,des1,img2,kp2,des2);
    return 0;
}

