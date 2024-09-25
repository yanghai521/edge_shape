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
		std::vector<Point*> template_points;          // 描述子中边缘点坐标集
		std::vector<double*> template_dxs;            // 描述子中边缘点X偏导集
		std::vector<double*> template_dys;            // 描述子中边缘点Y偏导集
	};

	ChipMatch(void);                                  //构造函数的声明
	~ChipMatch(void);                                 //析构函数的声明
	//Mat CreateTemplate(Mat &templateImage, double, double);    //声明创建模板的方法

	//模板图像高斯滤波处理
	Mat GaussImg(Mat& srcImage, int kernelsize, double sigma);
	//提取最佳边缘点集、生成模板信息描述子
	Mat EdgeImage(Mat& guassimage, double low, double max, Point** t_Points, double** t_derivativeX, double** t_derivativeY);
	//对特征直接旋转操作
	void RotateFeature(Point** t_Points, double** t_derivativeX, double** t_derivativeY, double leftAngle, double rightAngle);
	//搜索匹配过程
	double MatchTemplate(Mat& searchImage, double, double, Point& resultPoint0, double& angle, Point** t_Points, double** t_derivativeX, double** t_derivativeY);
	//绘制匹配结果
	Mat Draw_contour(Mat& drawImage, Mat& srcImage, Point** t_Points, Point center, double s);
};