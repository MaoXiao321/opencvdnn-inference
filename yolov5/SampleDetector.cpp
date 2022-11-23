#include <iostream>
#include <fstream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "SampleDetector.hpp"
using namespace std;

const float anchors_640[3][6] = { {10.0,  13.0, 16.0,  30.0,  33.0,  23.0},
								 {30.0,  61.0, 62.0,  45.0,  59.0,  119.0},
								 {116.0, 90.0, 156.0, 198.0, 373.0, 326.0} };

const float anchors_1280[4][6] = { {19, 27, 44, 40, 38, 94},{96, 68, 86, 152, 180, 137},{140, 301, 303, 264, 238, 542},
					   {436, 615, 739, 380, 925, 792} };

SampleDetector::SampleDetector(double thresh, double nms, double hierThresh):
	 mNms(nms), mThresh(thresh), mHIERThresh(hierThresh) {
		 cout << "Current config: nms:" << mNms << ", thresh:" << mThresh
			 << ", HIERThresh:" << mHIERThresh;
 }
	 //this->confThreshold = confThreshold;
	 //this->objThreshold = objThreshold;
	 //this->nmsThreshold = nms;

 int endsWith(string s, string sub) {
	 //字符串从后往前搜索指定内容，返回索引值，如果没有就返回-1。当索引值等于s.length() - sub.length()时，说明以sub结尾
	 return s.rfind(sub) == (s.length() - sub.length()) ? 1 : 0;
 }

 int SampleDetector::init(string namesFile, string weightsFile) {
	 cout << "Loading model...";
	 detector = readNet(weightsFile); //用opencvdnn读入
	 cout << "done！" << endl;

	 cout << "Loading classes.." ;
	 ifstream ifs(namesFile);
	 string line;
	 while (getline(ifs, line)) class_names.push_back(line); //将所有类别存到class_names中
	 
	 num_class = class_names.size();
	 if (num_class == 0) {
		 cout << "Failed getting labels from `" << namesFile << "`!";
		 return SampleDetector::ERROR_INVALID_INIT_ARGS;
	 }
	 cout << "类别数："<< num_class << endl;

	 if (endsWith(weightsFile, "6.onnx"))
	 {
		 anchors = (float*)anchors_1280; //配置参数
		 num_stride = 4;
		 inpHeight = 1280;
		 inpWidth = 1280;
	 }
	 else
	 {
		 anchors = (float*)anchors_640;
		 num_stride = 3;
		 inpHeight = 640;
		 inpWidth = 640;
	 }

	 return SampleDetector::INIT_OK;
 }

 void SampleDetector::unInit() {
 }

 Mat SampleDetector::resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left)
 {
	 //指针是一个变量，用来存地址。指针前再加个*是解引用，拿出这个地址存的值
	 int srch = srcimg.rows, srcw = srcimg.cols;
	 *newh = inpHeight; //newh是指针，指针前加*拿出该位置的值，这里是将该位置的值赋为inpHeight
	 *neww = inpWidth;

	 Mat dstimg;
	 if (keep_ratio && srch != srcw) {
		 float hw_scale = (float)srch / srcw; //高宽比
		 //调整图片大小
		 if (hw_scale > 1) {
			 *newh = inpHeight;
			 *neww = int(inpWidth / hw_scale);
			 resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			 *left = int((inpWidth - *neww) * 0.5);
			 copyMakeBorder(dstimg, dstimg, 0, 0, *left, inpWidth - *neww - *left, BORDER_CONSTANT, 114);
		 }
		 else {
			 *newh = (int)inpHeight * hw_scale;
			 *neww = inpWidth;
			 resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			 *top = (int)(inpHeight - *newh) * 0.5;
			 copyMakeBorder(dstimg, dstimg, *top, inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 114);
		 }
	 }
	 else {
		 resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	 }
	 return dstimg;
 }

 void SampleDetector::drawPred(float conf, int left, int top, int right, int bottom, const Mat& frame, int classid)   // Draw the predicted bounding box
 {
	 //Draw a rectangle displaying the bounding box
	 rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 2);

	 //Get the label for the class name and its confidence
	 string label = format("%.2f", conf);
	 label = class_names[classid] + ":" + label;

	 //Display the label at the top of the bounding box
	 int baseLine;
	 Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	 top = max(top, labelSize.height);
	 //rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
	 putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
 }

 int SampleDetector::detect(const Mat& frame, vector<Object>& result)
 {
	 if (frame.empty()) {
		 cout << "Invalid input!";
		 return ERROR_INVALID_INPUT;
	 }
	 int newh = 0, neww = 0, padh = 0, padw = 0;
	 Mat dstimg = resize_image(frame, &newh, &neww, &padh, &padw); //定义时是指针，指针要存入地址，因此这里用&。这里是地址传递，会改变实参。
	 
	 cout << "宽*高："<< inpWidth << "*" << inpHeight << endl;
	 Mat blob = blobFromImage(dstimg, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);
	 this->detector.setInput(blob);
	 
	 vector<Mat> outs;
	 this->detector.forward(outs, this->detector.getUnconnectedOutLayersNames());
	 int num_proposal = outs[0].size[1];
	 int nout = outs[0].size[2];
	 if (outs[0].dims > 2)
	 {
		 outs[0] = outs[0].reshape(0, num_proposal);
	 }
	 
	 /////generate proposals
	 vector<float> confidences;
	 vector<Rect> boxes;
	 vector<int> classIds;
	 float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;
	 int n = 0, q = 0, i = 0, j = 0, row_ind = 0; ///xmin,ymin,xamx,ymax,box_score,class_score
	 float* pdata = (float*)outs[0].data;
	 for (n = 0; n < num_stride; n++)   //
	 {
		 const float stride = pow(2, n + 3); //计算当前的stride
		 int num_grid_x = (int)ceil((inpWidth / stride)); //算特征图大小
		 int num_grid_y = (int)ceil((inpHeight / stride));
		 for (q = 0; q < 3; q++)    ///anchor
		 {
			 const float anchor_w = anchors[n * 6 + q * 2]; //拿到anhor大小
			 const float anchor_h = anchors[n * 6 + q * 2 + 1];
			 for (i = 0; i < num_grid_y; i++)
			 {
				 for (j = 0; j < num_grid_x; j++)
				 {
					 float box_score = pdata[4]; //遍历特征图，拿到置信度
					 if (box_score > mThresh)
					 {
						 Mat scores = outs[0].row(row_ind).colRange(5, nout);
						 Point classIdPoint;
						 double max_class_socre;
						 // Get the value and location of the maximum score
						 minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
						 max_class_socre *= box_score; //置信度*类别得分
						 if (max_class_socre > mHIERThresh)
						 {
							 const int class_idx = classIdPoint.x;
							 float cx = (pdata[0] * 2.f - 0.5f + j) * stride;  ///cx
							 float cy = (pdata[1] * 2.f - 0.5f + i) * stride;   ///cy
							 float w = powf(pdata[2] * 2.f, 2.f) * anchor_w;   ///w
							 float h = powf(pdata[3] * 2.f, 2.f) * anchor_h;  ///h

							 int left = int((cx - padw - 0.5 * w) * ratiow);
							 int top = int((cy - padh - 0.5 * h) * ratioh);

							 confidences.push_back((float)max_class_socre);
							 boxes.push_back(Rect(left, top, (int)(w * ratiow), (int)(h * ratioh)));
							 classIds.push_back(class_idx);
						 }
					 }
					 row_ind++;
					 pdata += nout;
				 }
			 }
		 }
	 }

	 // Perform non maximum suppression to eliminate redundant overlapping boxes with
	 // lower confidences
	 vector<int> indices;
	 NMSBoxes(boxes, confidences, mThresh, mNms, indices);
	 for (size_t i = 0; i < indices.size(); ++i)
	 {
		 int idx = indices[i];
		 Rect box = boxes[idx];
		 result.push_back({confidences[idx], class_names[classIds[idx]], box});

		 drawPred(confidences[idx], box.x, box.y,
			 box.x + box.width, box.y + box.height, frame, classIds[idx]);
	 }
	 return SampleDetector::PROCESS_OK;
 }