#include "opencv2/opencv.hpp"
#include<opencv2/xfeatures2d.hpp>
using namespace cv;
using namespace cv::xfeatures2d;

int thre = 40;
Mat src;
void trackBar(int, void*);

int main(int argc, char** argv)
{
    src = imread("/home/gxf/slam/BASD_voc_create/voc_create/src/000001.png"); 
    if (src.empty())
    {
        printf("can not load image \n");
        return -1;
    }
    namedWindow("input",WINDOW_AUTOSIZE);
    imshow("input", src);  

    namedWindow("output",WINDOW_AUTOSIZE);
    createTrackbar("threshould", "output", &thre,255, trackBar);
    cvWaitKey(0);  
    return 0;
}

void trackBar(int, void*)
{
    std::vector<KeyPoint> keypoints;
    Mat dst = src.clone();
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(thre);
    detector->detect(src,keypoints);
    drawKeypoints(dst, keypoints, dst, Scalar::all(-1), DrawMatchesFlags::DRAW_OVER_OUTIMG);  
    imshow("output", dst);  
}
