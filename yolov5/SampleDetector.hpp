#ifndef JI_SAMPLEDETECTOR_HPP
#define JI_SAMPLEDETECTOR_HPP

#include <string>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

class SampleDetector {

public:
    typedef struct {
        float prob;
        std::string name;
        cv::Rect rect;
    } Object;
    typedef struct {
        // 算法配置可选的配置参数
        double nms;
        double thresh;
        double hierThresh;
    } ALGO_CONFIG_TYPE;
    SampleDetector(double thresh, double nms, double hierThresh);
    //int init(const char* namesFile,const char* weightsFile);
    int init(string namesFile, string weightsFile);
    void unInit();
    int detect(const Mat& frame, vector<Object>& result);

public:
    static const int ERROR_BASE = 0x0100;
    static const int ERROR_INVALID_INPUT = 0x0101;
    static const int ERROR_INVALID_INIT_ARGS = 0x0102;
    static const int PROCESS_OK = 0x1001;
    static const int INIT_OK = 0x1002;

private:
    Net detector;
    void drawPred(float conf, int left, int top, int right, int bottom, const Mat& frame, int classid);
    vector<string> class_names;
    int num_class;

    int inpWidth;
    int inpHeight;
    float* anchors;
    int num_stride;

    double mThresh;
    double mHIERThresh; //置信度*类别得分
    double mNms;
    const bool keep_ratio = true;
    
    Mat resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left);
};

#endif