#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <string>
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

using namespace cv;
using namespace std;

inline int bainaryDesc(float x) { return (x > 0.0) ? 1 : 0; }

//将每张图片的绝对路径保存到vstrImageFilenames
void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while (!fTimes.eof())
    {
        string s;
        getline(fTimes, s);
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";

    int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for (int i = 0; i < nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}

void LoadImages_several_kitti(const string &strPathToSequence, vector<string> &vstrImageFilenames)
{
    vector<double> vTimestamps;
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while (!fTimes.eof())
    {
        string s;
        getline(fTimes, s);
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";

    int nTimes = vTimestamps.size();

    for (int i = 0; i < nTimes; i = i + 3) //每个kitti数据集隔两张取一张
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        string path = strPrefixLeft + ss.str() + ".png";
        vstrImageFilenames.push_back(path);
    }
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
    //cout << "keypoint size=" << keypoints.size() << endl;
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
    memcpy(descriptors.data, output.data_ptr(), (turn)*256 * sizeof(float)); //将outpuu中的数字复制给descriptors
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    static int count=0;
    ++count;
    cout << "  generate ASD time = " << ttrack <<"  "<<count<<endl;
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
        desA_ = (vector<float>)taf.reshape(1, 1);        //Mat::reshape(int cn, int rows=0) const， cn表示通道数，为0则表示不变。rows表示后面得到的行数
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
    //cout << "ASD2BASD time = " << ttrack << endl;
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

int main(int argc, char **argv)
{
    //加载图片
    vector<string> vstrImageFilenames; //利用LoadImage函数将所有的图片和时间戳加载到两个vector里面
    vector<double> vTimestamps;
    //加载一个数据集
    //string img_path = "/home/gxf/slam/slam_dataset/kitti/dataset3/sequences/00";
    //LoadImages(img_path, vstrImageFilenames, vTimestamps);

    //加载多个数据集
    vector<string> img_paths;
    img_paths.push_back("/media/gxf/Elements/opensource_dataset/KITTI/kitti/dataset_image/sequences/00");
    img_paths.push_back("/media/gxf/Elements/opensource_dataset/KITTI/kitti/dataset_image/sequences/01");
    img_paths.push_back("/media/gxf/Elements/opensource_dataset/KITTI/kitti/dataset_image/sequences/02");
    img_paths.push_back("/media/gxf/Elements/opensource_dataset/KITTI/kitti/dataset_image/sequences/03");
    img_paths.push_back("/media/gxf/Elements/opensource_dataset/KITTI/kitti/dataset_image/sequences/04");
    img_paths.push_back("/media/gxf/Elements/opensource_dataset/KITTI/kitti/dataset_image/sequences/05");
    img_paths.push_back("/media/gxf/Elements/opensource_dataset/KITTI/kitti/dataset_image/sequences/06");
    img_paths.push_back("/media/gxf/Elements/opensource_dataset/KITTI/kitti/dataset_image/sequences/07");
    img_paths.push_back("/media/gxf/Elements/opensource_dataset/KITTI/kitti/dataset_image/sequences/08");
    img_paths.push_back("/media/gxf/Elements/opensource_dataset/KITTI/kitti/dataset_image/sequences/09");
    img_paths.push_back("/media/gxf/Elements/opensource_dataset/KITTI/kitti/dataset_image/sequences/10");
    img_paths.push_back("/media/gxf/Elements/opensource_dataset/KITTI/kitti/dataset_image/sequences/11");
    img_paths.push_back("/media/gxf/Elements/opensource_dataset/KITTI/kitti/dataset_image/sequences/12");
    img_paths.push_back("/media/gxf/Elements/opensource_dataset/KITTI/kitti/dataset_image/sequences/13");
    img_paths.push_back("/media/gxf/Elements/opensource_dataset/KITTI/kitti/dataset_image/sequences/14");
    img_paths.push_back("/media/gxf/Elements/opensource_dataset/KITTI/kitti/dataset_image/sequences/15");
    img_paths.push_back("/media/gxf/Elements/opensource_dataset/KITTI/kitti/dataset_image/sequences/16");
    img_paths.push_back("/media/gxf/Elements/opensource_dataset/KITTI/kitti/dataset_image/sequences/17");
    for (int i = 0; i < img_paths.size(); ++i)
        LoadImages_several_kitti(img_paths[i], vstrImageFilenames);
    int nImages = vstrImageFilenames.size();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl
         << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl
         << endl;

    //加载模型
    torch::jit::script::Module module = torch::jit::load("/home/gxf/slam/ORB-SlAM2/ASDmodule/ASDNet.pt");
    assert(module != nullptr);
    cout << "module loaded" << endl;
    module.to(at::kCUDA);

    // 读取图片提取描述子
    cv::Mat im;
    vector<Mat> descriptors;
    for (int ni = 0; ni < nImages; ni++)
    {
        // Read image from file
        im = cv::imread(vstrImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);

        /*if (ni >= 2000)
        {
            break;
        }*/
        if (im.empty())
        {
            cerr << endl
                 << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }
        Ptr<ORB> detector = ORB::create(1500);
        vector<cv::KeyPoint> kp1;
        cv::Mat BASD1;
        detector->detectAndCompute(im, noArray(), kp1, BASD1);
        get_BASD(im, kp1, BASD1, module);

        descriptors.push_back(BASD1);
        //cout << "extracting features from image " << index++ << endl;
    }
    cout << "extract total " << descriptors.size() * 1500 << " features." << endl;

    // create vocabulary
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    cout << "creating vocabulary, please wait ... " << endl;
    DBoW3::Vocabulary vocab(10,6);
    vocab.create(descriptors);
    cout << "vocabulary info: " << vocab << endl;
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    cout << "voc create time = " << ttrack << endl;
    vocab.save("vocab_larger.yml.gz");
    cout << "done" << endl;

    return 0;
}