#include <opencv2/opencv.hpp>
#include <vector>
#include <list>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <chrono>

using namespace std;
using namespace cv;
inline int bainaryDesc(float x) { return (x > 0.0) ? 1 : 0; }

//加载torch模型
//BASD是引用传递，
void get_BASD(const cv::Mat &image, vector<cv::KeyPoint> &keypoints, cv::Mat &BASD, torch::jit::script::Module module);
//输入参数：图片，关键点，描述子，module
//得到描述子
void computeSIFTDescriptors(const Mat &image, vector<KeyPoint> &keypoints, Mat &descriptors, torch::jit::script::Module module);

void computeBASDfromASD(const Mat &ASD, Mat &BASD); //将256的ASD转换成ASD

bool refineMatchesWithHomography(
	const std::vector<cv::KeyPoint> &queryKeypoints,
	const std::vector<cv::KeyPoint> &trainKeypoints,
	float reprojectionThreshold, std::vector<cv::DMatch> &matches,
	cv::Mat &homography);

void draw_good_match(cv::Mat &img1, const vector<KeyPoint> &kp1, cv::Mat &des1, cv::Mat &img2, const vector<KeyPoint> &kp2, cv::Mat &des2);
int main(int argc, char **argv)
{
	torch::jit::script::Module module = torch::jit::load("/home/gxf/slam/ORB-SlAM2/ASDmodule/ASDNet.pt");
	assert(module != nullptr);
	cout << "module loaded" << endl;
	module.to(at::kCUDA);
	while (1)
		{
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

	cv::Mat img1 = cv::imread("/home/gxf/slam/BASD_voc_create/voc_create/src/000092.png", 0);
	cv::Mat img2 = cv::imread("/home/gxf/slam/BASD_voc_create/voc_create/src/000095.png", 0);

	Ptr<ORB> detector = ORB::create(1500);
	vector<cv::KeyPoint> kp1;
	cv::Mat BASD1;
	detector->detectAndCompute(img1, noArray(), kp1, BASD1);
	get_BASD(img1, kp1, BASD1, module);

	vector<cv::KeyPoint> kp2;
	cv::Mat BASD2;
	detector->detectAndCompute(img2, noArray(), kp2, BASD2);
	get_BASD(img2, kp2, BASD2, module);

	std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
	double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
	//cout << "time = " << ttrack << endl;
		}
	//draw_good_match(img1, kp1, BASD1, img2, kp2, BASD2);
	return 0;
}

//BASD是引用传递，
void get_BASD(const cv::Mat &image, vector<cv::KeyPoint> &keypoints, cv::Mat &BASD, torch::jit::script::Module module)
{
	cv::Mat ASD;
	int nkeypoints = keypoints.size();
	ASD.create(nkeypoints, 256, CV_32FC1);
	computeSIFTDescriptors(image, keypoints, ASD, module);
	//BASD.create(nkeypoints, 32, CV_8U);
	computeBASDfromASD(ASD, BASD); //将256的ASD转换成ASD
}

//输入参数：图片，关键点，描述子，module
//得到描述子
void computeSIFTDescriptors(const cv::Mat &image, vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors, torch::jit::script::Module module)
{
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	cv::Mat img_float;
	int turn = 1;
	int width = image.cols;
	int height = image.rows;
	cout << "keypoint size=" << keypoints.size() << endl;
	for (size_t i = 0; i < keypoints.size(); i++)
	{
		int x = cvRound(keypoints[i].pt.x);
		int y = cvRound(keypoints[i].pt.y);
		if (x - 16 > 0 && x + 16 < width && y - 16 > 0 && y + 16 < height)
		{
			Mat patch = image(Rect(x - 16, y - 16, 32, 32)); // x  ,  y  ,  width  ,  height
			if (turn == 1)
			{
				img_float = patch;
			} //还不能用vconcat
			else
			{
				vconcat(img_float, patch, img_float); //将前面两个矩阵垂直叠起来，输出到第三个参数
			}
			turn++;
		}
	}
	turn--; //因为初始值为1，所以这里要减1才是真正的patch数量

	img_float.convertTo(img_float, CV_32F, 1.0 / 255);
	std::vector<torch::jit::IValue> inputs;
	auto img_tensor = torch::from_blob(img_float.data, {turn, 32, 32, 1}).permute({0, 3, 1, 2}); //将img_float.data叠成{turn ,32, 32, 1}的矩阵
																								 //permute，tensor维度换位，这里换成{turn,1,32,32}
	inputs.push_back(img_tensor.to(torch::kCUDA));
	at::Tensor output = module.forward(inputs).toTensor();
	output = output.to(torch::kCPU);
		std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
	double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
cout << "generate ASD time = " << ttrack << endl;

std::chrono::steady_clock::time_point t11 = std::chrono::steady_clock::now();
	memcpy(descriptors.data, output.data_ptr(), (turn)*256 * sizeof(float)); //将outpuu中的数字复制给descriptors
	std::chrono::steady_clock::time_point t22 = std::chrono::steady_clock::now();
	double ttrack1 = std::chrono::duration_cast<std::chrono::duration<double>>(t22 - t11).count();
	cout << "copy ASD time = " << ttrack1 << endl;
}

//ASD row= 4002  col= 256
//BASD row= 4002  col= 32
void computeBASDfromASD(const Mat &ASD, Mat &BASD) //将256的ASD转换成ASD
{

	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	for (int i = 0; i < ASD.rows; i++)
	{
		cv::Mat taf;
		vector<float> desA_;
		ASD.rowRange(i, i + 1).convertTo(taf, CV_32FC1); // 取出特定行，包括左边界，但不包括右边界
														 //cout<<"taf row= "<<taf.rows<<"  col="<<taf.cols<<endl;//1 256
														 //cout<<"calculate 1"<<endl;
		desA_ = (vector<float>)taf.reshape(1, 1);		 //Mat::reshape(int cn, int rows=0) const， cn表示通道数，为0则表示不变。rows表示后面得到的行数
														 //cout<<"calculate 2"<<endl;

		//cv::Mat desA_;
		//desA_=ASD.rowRange(i,i);
		for (int j = 0, k = 0; j < 256 && k < 32; j += 8, ++k)
		{
			int val, t;
			t = bainaryDesc(desA_[j + 0]);
			val = t;

			t = bainaryDesc(desA_[j + 1]);
			val |= t << 1;
			t = bainaryDesc(desA_[j + 2]);
			val |= t << 2;
			t = bainaryDesc(desA_[j + 3]);
			val |= t << 3;
			t = bainaryDesc(desA_[j + 4]);
			val |= t << 4;
			t = bainaryDesc(desA_[j + 5]);
			val |= t << 5;
			t = bainaryDesc(desA_[j + 6]);
			val |= t << 6;
			t = bainaryDesc(desA_[j + 7]);
			val |= t << 7;
			BASD.ptr(i)[k] = (uchar)val;
			if (i % 2000 == 0)
			{
				//cout<<"  "<<int(BASD.ptr(i)[k] ) ;
			}
		}
	}
	std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
	double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
	cout << "ASD2BASD time = " << ttrack << endl;
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

void draw_good_match(cv::Mat &img1, const vector<KeyPoint> &kp1, cv::Mat &des1, cv::Mat &img2, const vector<KeyPoint> &kp2, cv::Mat &des2)
{
	vector<DMatch> matches;
	BFMatcher bfMatcher(NORM_HAMMING);
	bfMatcher.match(des1, des2, matches);

	Mat matHomo;
	refineMatchesWithHomography(kp1, kp2, 4.0, matches, matHomo);
	cout << "[Info] Homography T : " << matHomo << endl;
	cout << "[Info] # of matches : " << matches.size() << endl;

	Mat imResult;
	drawMatches(img1, kp1, img2, kp2, matches, imResult);

	imshow("refined matches", imResult);
	imwrite("refined_BASD_matches92_95.jpg", imResult);
	waitKey();
}
