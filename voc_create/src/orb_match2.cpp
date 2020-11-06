#include <iostream>
#include <fstream>
#include <sstream>

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

void KeyPointsToPoints(vector<KeyPoint> kpts, vector<Point2f> &pts);

bool refineMatchesWithHomography(
    const std::vector<cv::KeyPoint> &queryKeypoints,
    const std::vector<cv::KeyPoint> &trainKeypoints,
    float reprojectionThreshold, std::vector<cv::DMatch> &matches,
    cv::Mat &homography);

/** @function main */
int main(int argc, char *argv[])
{

    /************************************************************************/
    /* 特征点检测，特征提取，特征匹配，计算投影变换                            */
    /************************************************************************/

    // 读取图片
    Mat img1Ori = imread("1.jpg", 0);
    Mat img2Ori = imread("2.jpg", 0);

    // 缩小尺度
    Mat img1, img2;
    resize(img1Ori, img1, Size(img1Ori.cols / 4, img1Ori.cols / 4));
    resize(img2Ori, img2, Size(img2Ori.cols / 4, img2Ori.cols / 4));

    Ptr<ORB> detector = ORB::create();                  // 创建orb特征点检测
    cv::Ptr<cv::DescriptorExtractor> extractor =  FAST::create();      // 用Freak特征来描述特征点
    cv::Ptr<cv::DescriptorMatcher> matcher =  cv::BFMatcher(cv::NORM_HAMMING, // 特征匹配，计算Hamming距离
                                                               true);

    vector<KeyPoint> keypoints1; // 用于保存图中的特征点
    vector<KeyPoint> keypoints2;
    Mat descriptors1; // 用于保存图中的特征点的特征描述
    Mat descriptors2;

    detector->detect(img1, keypoints1); // 检测第一张图中的特征点
    detector->detect(img2, keypoints2);

    extractor->compute(img1, keypoints1, descriptors1); // 计算图中特征点位置的特征描述
    extractor->compute(img2, keypoints2, descriptors2);

    vector<DMatch> matches;
    matcher->match(descriptors1, descriptors2, matches);

    Mat imResultOri;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, imResultOri,
                CV_RGB(0, 255, 0), CV_RGB(0, 255, 0));
    cout << "[Info] # of matches : " << matches.size() << endl;

    Mat matHomo;
    refineMatchesWithHomography(keypoints1, keypoints2, 3, matches, matHomo);
    cout << "[Info] Homography T : " << matHomo << endl;
    cout << "[Info] # of matches : " << matches.size() << endl;

    Mat imResult;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, imResult,
                CV_RGB(0, 255, 0), CV_RGB(0, 255, 0));

    // 计算光流
    vector<uchar> vstatus;
    vector<float> verrs;
    vector<Point2f> points1;
    vector<Point2f> points2;
    KeyPointsToPoints(keypoints1, points1);

    calcOpticalFlowPyrLK(img1, img2, points1, points2, vstatus, verrs);

    Mat imOFKL = img1.clone();
    for (int i = 0; i < vstatus.size(); i++)
    {
        if (vstatus[i] && verrs[i] < 15)
        {
            line(imOFKL, points1[i], points2[i], CV_RGB(255, 255, 255), 1, 8, 0);
            circle(imOFKL, points2[i], 3, CV_RGB(255, 255, 255), 1, 8, 0);
        }
    }

    imwrite("opt.jpg", imOFKL);
    imwrite("re1.jpg", imResultOri);
    imwrite("re2.jpg", imResult);

    imshow("Optical Flow", imOFKL);
    imshow("origin matches", imResultOri);
    imshow("refined matches", imResult);
    waitKey();

    return -1;
}

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

void KeyPointsToPoints(vector<KeyPoint> kpts, vector<Point2f> &pts)
{
    for (int i = 0; i < kpts.size(); i++)
    {
        pts.push_back(kpts[i].pt);
    }

    return;
}
