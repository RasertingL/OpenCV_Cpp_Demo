#include <vector>
#include <iostream>
#include <io.h>
#include<opencv2/opencv.hpp>   //基础功能
#include <opencv2/core/core.hpp>  //核心组件
#include <opencv2/highgui/highgui.hpp>  //GUI 
#include <opencv2/imgproc/imgproc.hpp>  //图像处理
#include <opencv2/features2d/features2d.hpp> //特征检测SurfFeatureDetector要用
//下面实现一个简单的基于直方图比较的文件夹漫步图片查找demo

using namespace cv;
using namespace std;

/*
直方图是图像像素值的统计学特征，计算代价较小，具有图像平移，旋转，缩放不变的特征。广泛应用于图像书里
和各个领域，特别是灰度图的阈值分割，基于颜色的图像检索以及图像分类，反向投影跟踪。

常见分为：
	灰度直方图
	颜色直方图

Bins指的是直方图的大小范围，对像素0-255之间，最少有256个bins，此外还有16,32,48,128等。256除以bin的大小应该是整数倍。

*/

/*
作用：
	可以用直方图的这个统计特征来二值分割图像。
	进行图像的检索，
	相似图像的分类
	反向投影
*/

//读取文件目录函数
void getFiles(string path, vector<string>& files)
{
	//文件句柄
	intptr_t   hFile = 0;
	//文件信息
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之
			//如果不是,加入列表
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}


int main() {

	Mat src1 = imread("D:\\Pictures\\Blender_Suzanne1.jpg");
	Mat src2;
	Mat dst;

	imshow("imput1", src1);
	

	Mat hsv1, hsv2;

	cvtColor(src1, hsv1, COLOR_BGR2HSV);

	//对于h通道和s通道我们给Bins分别为60和64
	int h_bins = 60; 
	int s_bins = 64;

	const int histSize[] = { h_bins, s_bins };
	float h_ranges[] = { 0 ,180 };
	float s_ranges[] = { 0, 256 };
	const float* ranges_c[] = { h_ranges, s_ranges };

	int channels2[] = { 0, 1 };

	double dst_src1_sc2 = 9999;

	Mat hist1, hist2;
	calcHist(&hsv1, 1, channels2, Mat(), hist1, 2, histSize, ranges_c);

	normalize(hist1, hist1, 0, 1, NORM_MINMAX, -1, Mat());

	//进行文件目录读取
	const char * filePath = "D:\\Pictures";
	vector<string> files;

	////获取该路径下的所有文件
	getFiles(filePath, files);

	char str[30];
	int size = files.size();
	for (int i = 0; i < size; i++)
	{
		src2 = imread(files[i].c_str());

		if (src2.empty())
			continue;

		cvtColor(src2, hsv2, COLOR_BGR2HSV);
		calcHist(&hsv2, 1, channels2, Mat(), hist2, 2, histSize, ranges_c);
		normalize(hist2, hist2, 0, 1, NORM_MINMAX, -1, Mat());

		double src1_src2 = compareHist(hist1, hist2, 3);

		if (src1_src2 < dst_src1_sc2) {
			dst_src1_sc2 = src1_src2;
			dst = src2.clone();

			cout << files[i].c_str() << endl;
			cout << src1_src2 << "         " << dst_src1_sc2 << endl;
			cout << endl;
		}
	}

	imshow("dst", dst);

	waitKey(0);

}


