#include <iostream>
#include "shape.h"

using namespace std;
using namespace cv;
int main(int argc, char** argv) {
	//��ȡģ��ͼ���Ŀ��ͼ��
	Mat src = imread("G:\\template\\chip\\t\\text.png", 0);
	Mat dst = imread("G:\\template\\chip-picture\\samples\\01.png", 0);
	//ģ����Ϣ��������ز���
	double lowThreshold = 60;
	double hightThreshold = 128;
	double min_score = 0.9;
	double greediness = 0.9;

	Point center;

	//����Ƿ�ɹ�����ͼ��
	if (src.empty()) {
		cout << "����:û�м���ģ��ͼ��" << endl;
		return -1;
	}
	if (dst.empty()) {
		cout << "����:û�м���Ŀ��ͼ��" << endl;
		return -1;
	}
	//��ʾģ��ͼ��
	namedWindow("src", WINDOW_AUTOSIZE);
	imshow("src", src);
	//��ʾĿ��ͼ��
	namedWindow("dst", WINDOW_AUTOSIZE);
	imshow("dst", dst);
	//ʵ����
	ChipMatch chipmatch;

	//---------------��˹����---------------//
	int kernelsize = 7;
	double sigma = 1.6;
	Mat src0 = chipmatch.GaussImg(src, kernelsize, sigma);
	/*namedWindow("yuan0", WINDOW_AUTOSIZE);
	imshow("yuan0", src0);*/
	//ƥ����Ϣ�ṹ�壺����Ϊģ����Ϣ�������б�Ե���������ꡢX��Y����Ĺ�һ���ݶȷ�ֵ
	struct Edge_Data {
		// �������б�Ե�����꼯
		std::vector<Point*> edge_points;
		// �������б�Ե��Xƫ����
		std::vector<double*> edge_dxs;
		// �������б�Ե��Yƫ����
		std::vector<double*> edge_dys;
	};
	//ʵ������Ե���ݽṹ��
	Edge_Data edge_data;
	//��ʼ����ָ��
	Point* edge_point = nullptr;
	double* edge_dx = nullptr;
	double* edge_dy = nullptr;

	//-----------------------ģ����Ϣ����--------------------//

	


	//����ģ��ͼ������ƥ��㡢���ƶȵ÷֡�ƫ�ƽǶ�
	Point t;
	double match_score;
	double angle;

	//-----------------------����ƥ�����--------------------//

	clock_t start_time = clock();
	match_score = chipmatch.MatchTemplate(dst, min_score, greediness, t, angle, &edge_point, &edge_dx, &edge_dy);
	clock_t end_time = clock();
	//��ȡ����ƥ����̵�ʱ��
	double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
	//���ƥ�����ƶȡ�������ƥ��ʱ�䡢ƫ�ƽǶ�
	

	//-----------------------��ʾƥ����----------------------//

	
	//�ͷŶ����ָ���ַ
	delete[] edge_point;
	delete[] edge_dx;
	delete[] edge_dy;

	waitKey(0);
	destroyAllWindows();

	return 0;
}