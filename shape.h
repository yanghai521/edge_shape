#pragma once

#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <vector>
#include<time.h>
#include<thread>
using namespace cv;
using namespace std;
class ChipMatch
{
private:
	int candidatePoint_n;
	int modelWidth;
	int modelHeight;
	Point** Points;
	double** derivativeX;
	double** derivativeY;

	Point centerPoint;
	int pyramid_n;

	Point** t_Points;
	double** t_derivativeX;
	double** t_derivativeY;

	Point* newPoints;
	double* newDerivativeX;
	double* newDerivativeY;



	struct TemplateData {
		std::vector<std::vector<Point2f>> template_points;
		std::vector<std::vector<double>> template_dxs;
		std::vector<std::vector<double>> template_dys;
	};
	
	TemplateData templatedata;

	void CreateMatrix(double**& matrix, int, int);
	void ReleaseMatrix(double**& matrix, int size);


public:
	struct TemplateData0 {
		std::vector<Point*> template_points;          // �������б�Ե�����꼯
		std::vector<double*> template_dxs;            // �������б�Ե��Xƫ����
		std::vector<double*> template_dys;            // �������б�Ե��Yƫ����
	};

	ChipMatch(void);                                  //���캯��������
	~ChipMatch(void);                                 //��������������
	//Mat CreateTemplate(Mat &templateImage, double, double);    //��������ģ��ķ���

	//ģ��ͼ���˹�˲�����
	Mat GaussImg(Mat& srcImage, int kernelsize, double sigma);
	//��ȡ��ѱ�Ե�㼯������ģ����Ϣ������
	Mat EdgeImage(Mat& guassimage, double low, double max, Point** t_Points, double** t_derivativeX, double** t_derivativeY);
	//������ֱ����ת����
	void RotateFeature(Point** t_Points, double** t_derivativeX, double** t_derivativeY, double leftAngle, double rightAngle);
	//����ƥ�����
	double MatchTemplate(Mat& searchImage, double, double, Point& resultPoint0, double& angle, Point** t_Points, double** t_derivativeX, double** t_derivativeY);
	//����ƥ����
	Mat Draw_contour(Mat& drawImage, Mat& srcImage, Point** t_Points, Point center, double s);
};