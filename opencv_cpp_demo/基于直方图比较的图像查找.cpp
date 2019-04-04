#include <vector>
#include <iostream>
#include <io.h>
#include<opencv2/opencv.hpp>   //��������
#include <opencv2/core/core.hpp>  //�������
#include <opencv2/highgui/highgui.hpp>  //GUI 
#include <opencv2/imgproc/imgproc.hpp>  //ͼ����
#include <opencv2/features2d/features2d.hpp> //�������SurfFeatureDetectorҪ��
//����ʵ��һ���򵥵Ļ���ֱ��ͼ�Ƚϵ��ļ�������ͼƬ����demo

using namespace cv;
using namespace std;

/*
ֱ��ͼ��ͼ������ֵ��ͳ��ѧ������������۽�С������ͼ��ƽ�ƣ���ת�����Ų�����������㷺Ӧ����ͼ������
�͸��������ر��ǻҶ�ͼ����ֵ�ָ������ɫ��ͼ������Լ�ͼ����࣬����ͶӰ���١�

������Ϊ��
	�Ҷ�ֱ��ͼ
	��ɫֱ��ͼ

Binsָ����ֱ��ͼ�Ĵ�С��Χ��������0-255֮�䣬������256��bins�����⻹��16,32,48,128�ȡ�256����bin�Ĵ�СӦ������������

*/

/*
���ã�
	������ֱ��ͼ�����ͳ����������ֵ�ָ�ͼ��
	����ͼ��ļ�����
	����ͼ��ķ���
	����ͶӰ
*/

//��ȡ�ļ�Ŀ¼����
void getFiles(string path, vector<string>& files)
{
	//�ļ����
	intptr_t   hFile = 0;
	//�ļ���Ϣ
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//�����Ŀ¼,����֮
			//�������,�����б�
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

	//����hͨ����sͨ�����Ǹ�Bins�ֱ�Ϊ60��64
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

	//�����ļ�Ŀ¼��ȡ
	const char * filePath = "D:\\Pictures";
	vector<string> files;

	////��ȡ��·���µ������ļ�
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


