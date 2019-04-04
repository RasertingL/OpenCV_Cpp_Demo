/*
车道线检测，需要完成以下功能：

图像裁剪：通过设定图像ROI区域，拷贝图像获得裁剪图像
反透视变换：用的是室外采集到的视频，没有对应的变换矩阵。所以建立二维坐标，通过四点映射的方法计算矩阵，进行反透视变化。后因ROI区域的设置易造成变换矩阵获取困难和插值得到的透视图效果不理想，故没应用
二值化：先变化为灰度图，然后设定阈值直接变成二值化图像。
形态学滤波：对二值化图像进行膨胀。
边缘检测：canny变化、sobel变化和laplacian变化中选择了效果比较好的canny变化，三者在代码中均可以使用，canny变化效果稍微好一点。
直线检测：实现了两种方法 1>使用opencv库封装好的霍夫直线检测函数，在原图对应区域用红线描出车道线 2>自己写一种直线检测，在头文件中，遍历ROI区域进行特定角度范围的直线检测。两种方法均可在视频中体现，第一种方法运行效率较快。
然后用利用角度特征进行直线筛选，再对直线进行聚类，得到最终的两条线。
*/

#include<cstdio>
#include<cmath>
#include<iostream>
#include<vector>
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>				
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

bool test_time = true;
bool show_steps = false;
const double PI = 3.1415926535;

// m: 斜率, b: 与y轴的截距, norm: 线长
struct hough_pts {
	double m, b, norm;
	hough_pts(double m, double b, double norm) :
		m(m), b(b), norm(norm) {};
};

//对黄线进行增强
void yellowline(cv::Mat img, cv::Mat & result) {
	cv::Mat img_hsv;
	cv::cvtColor(img, img_hsv, cv::COLOR_RGB2HSV);

	//下面设定一个三通道的下限，一个三通道的上限
	cv::Scalar low_yellow = cv::Scalar(40, 100, 20);
	cv::Scalar up_yellow = cv::Scalar(100, 255, 255);

	cv::Mat mask;
	//进行二值化操作，在low 和up范围之内是255， 不然是0
	cv::inRange(img_hsv, low_yellow, up_yellow, mask);
	
	cv::Mat gray;
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

	cv::addWeighted(mask, 1.0, gray, 1.0, 1.0, result);

}

void getROI(cv::Mat img, std::vector<cv::Point2i> rectpoints, cv::Mat &masked) {
	cv::Mat mask = cv::Mat::zeros(img.size(), img.type());//整个变黑
	//然后把ROI区域变为白色(即接下来作为掩膜，把除了ROI区域以外的地方都变黑)
	cv::fillConvexPoly(mask, rectpoints, cv::Scalar(255)); 
	//对图像像素点进行与操作，前两个参数是融合的两个源图，最后一个参数是掩膜，第三个参数是输出参数
	cv::bitwise_and(img, img, masked, mask);
}

//画线函数
void drawLines(cv::Mat &img, std::vector<cv::Vec4i> lines,cv::Scalar color) {
	for (int i = 0; i < lines.size(); i++) {
		cv::Point p1 = cv::Point(lines[i][0], lines[i][1]);
		cv::Point p2 = cv::Point(lines[i][2], lines[i][3]);
		cv::line(img, p1, p2, color, 4, 8);
	}
}

//求Hough直线检测出的线的均值（左右车道蜂聚直线斜率判断）
void averageLines(cv::Mat, std::vector<cv::Vec4i> lines, double y_min, double y_max, std::vector<cv::Vec4i>& output) {
	std::vector<hough_pts> left_lane, right_lane;
	for (int i = 0; i < lines.size(); ++i) {
		cv::Point2f pt1 = cv::Point2f(lines[i][0], lines[i][1]);
		cv::Point2f pt2 = cv::Point2f(lines[i][2], lines[i][3]);
		//m是斜率，b是截距，norm是线长度
		//运用公式y=m * x + b ,   b= y - m * x
		double m = (pt2.y - pt1.y) / (pt2.x - pt1.x);
		double b = -m * pt1.x + pt1.y;
		double norm = sqrt((pt2.x - pt1.x)*(pt2.x - pt1.x) + (pt2.y - pt1.y)*(pt2.y - pt1.y));
		if (m < 0) { // left lane
			left_lane.push_back(hough_pts(m, b, norm));
		}
		if (m > 0) { // right lane
			right_lane.push_back(hough_pts(m, b, norm));
		}
	}

	double b_avg_left = 0.0, m_avg_left = 0.0, xmin_left, xmax_left;
	for (int i = 0; i < left_lane.size(); ++i) {
		b_avg_left += left_lane[i].b;
		m_avg_left += left_lane[i].m;
	}
	b_avg_left /= left_lane.size();
	m_avg_left /= left_lane.size();
	//公式  x = (y - b) / m 
	xmin_left = int((y_min - b_avg_left) / m_avg_left);
	xmax_left = int((y_max - b_avg_left) / m_avg_left);
	// left lane
	output.push_back(cv::Vec4i(xmin_left, y_min, xmax_left, y_max));

	double b_avg_right = 0.0, m_avg_right = 0.0, xmin_right, xmax_right;
	for (int i = 0; i < right_lane.size(); ++i) {
		b_avg_right += right_lane[i].b;
		m_avg_right += right_lane[i].m;
	}
	b_avg_right /= right_lane.size();
	m_avg_right /= right_lane.size();
	xmin_right = int((y_min - b_avg_right) / m_avg_right);
	xmax_right = int((y_max - b_avg_right) / m_avg_right);
	// right lane
	output.push_back(cv::Vec4i(xmin_right, y_min, xmax_right, y_max));
}

//角度滤波，除噪(让直线的角度控制在30°- 80°之间，因为车道从驾驶位置看，两条线一般的张开角度是这么大)
void bypassAngleFilter(std::vector<cv::Vec4i> lines, double low_thres, double high_thres, std::vector<cv::Vec4i>& output) {
	for (int i = 0; i < lines.size(); ++i) {
		double x1 = lines[i][0], y1 = lines[i][1];
		double x2 = lines[i][2], y2 = lines[i][3];
		double angle = abs(atan((y2 - y1) / (x2 - x1)) * 180 / PI);
		if (angle > low_thres && angle < high_thres) {
			output.push_back(cv::Vec4i(x1, y1, x2, y2));
		}
	}
}

//检测与判定函数
void judgefunction(cv::Mat img, std::vector<cv::Point2i>rectpoints, double low_thres, double high_thres, cv::Mat &img_all_lines) {
	cv::Mat gray;
	//进行黄色车道线增强
	yellowline(img, gray);//得到gray图像
	

	cv::Mat gray_blur;
	//进行高斯滤波
	cv::GaussianBlur(gray, gray_blur, cv::Size(3, 3), 0, 0);
	//进行膨胀操作
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::morphologyEx(gray_blur, gray_blur, cv::MORPH_DILATE, element, cv::Point(-1, -1), 3);
	//进行canny边缘检测
	cv::Mat edge;
	cv::Canny(gray_blur, edge, 50, 180);

	cv::Mat masked;
	getROI(edge, rectpoints, masked);

	//开始进行累计概率霍夫变换
	std::vector<cv::Vec4i> lines;
	cv::HoughLinesP(masked, lines, 1, CV_PI / 180, 26, 5, 50); 

	//直接把累计霍夫变换得到的线画出来
	cv::Mat rowline_img = cv::Mat::zeros(img.size(), CV_8UC3);
	drawLines(rowline_img, lines, cv::Scalar(255, 0, 0));

	//下面进行角度滤波
	std::vector<cv::Vec4i> filtered_lines;
	bypassAngleFilter(lines, low_thres, high_thres, filtered_lines);

	//然后进行左右车道蜂聚直线斜率判断
	cv::Mat avg_img = cv::Mat::zeros(img.size(), CV_8UC3);
	std::vector<cv::Vec4i> avg_lines;
	averageLines(img, filtered_lines, int(img.rows * 0.65), img.rows, avg_lines);
	drawLines(avg_img, avg_lines, cv::Scalar(255, 0, 0));

	if (test_time) {
		cv::imshow("正常图", img);
		cv::imshow("黄色车道线增强图", gray);
		cv::imshow("高斯滤波之后", gray_blur);
		cv::imshow("canny后", edge);
		cv::imshow("ROI后mask_img", masked);
		cv::imshow("直接绘制的直线", rowline_img);
		cv::imshow("avg之后绘制的直线", avg_img);

		cv::waitKey(0);
	}

	cv::Mat colorMask = cv::Mat::zeros(img.size(), CV_8UC3);
	std::vector<cv::Point2i> linevertices;
	linevertices.push_back(cv::Point2i(avg_lines[0][0], avg_lines[0][1]));
	linevertices.push_back(cv::Point2i(avg_lines[0][2], avg_lines[0][3]));
	linevertices.push_back(cv::Point2i(avg_lines[1][2], avg_lines[1][3]));
	linevertices.push_back(cv::Point2i(avg_lines[1][0], avg_lines[1][1]));

	cv::fillConvexPoly(colorMask, linevertices, cv::Scalar(0, 255, 0));

	cv::addWeighted(avg_img, 1.0, img, 1.0, 0, img_all_lines);
	cv::addWeighted(img_all_lines, 1.0, colorMask, 0.6, 0.0, img_all_lines);




}

//对每一帧进行处理
void ProcessFrame(cv::Mat img, cv::Mat &result, int h, int w) {
	cv::Size size = cv::Size(h, w);
	std::vector<cv::Point2i> rectpoints;
	rectpoints.push_back(cv::Point2i(200, size.width - 80));
	rectpoints.push_back(cv::Point2i(490, int(size.width*0.65)));
	rectpoints.push_back(cv::Point2i(820, int(size.width*0.65)));
	rectpoints.push_back(cv::Point2i(size.height - 150, size.width - 80));
	//rectpoints.push_back(cv::Point2i(100, size.width - 80));
	double low_theta = 30.0;  //最低角度（为了之后去除噪声直线）
	double high_theta = 80.0;  //最大角度
	//下面进入真正的检测函数
	judgefunction(img, rectpoints, low_theta, high_theta, result);

}


int main() {
	std::string dir_video = "test_videos/";
	std::string video_output = dir_video + "detection.avi";

	//读取视频
	cv::VideoCapture cap(dir_video + "test.mp4");
	//用来之后保存处理后的视频
	cv::VideoWriter writer;

	int fps = 30;
	int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	std::cout << width << "   " << height << std::endl; //输出为1280   720

	cv::Size size(width, height);

	writer.open(video_output, CV_FOURCC('M', 'J', 'P', 'G'), fps, size);
	while (1) {
		cv::Mat frame;
		cap >> frame;
		if (frame.empty()) {
			std::cout << "The end" << std::endl;
			break;
		}
		cv::Mat result;
		ProcessFrame(frame, result, height, width);
		writer << result;
		cv::imshow("result", result);
		cv::waitKey(30);
	}
	cap.release();
}