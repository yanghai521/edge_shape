#include <iostream>
#include "shape.h"

using namespace std;
using namespace cv;
int main(int argc, char** argv) {
	//读取模板图像和目标图像
	Mat src = imread("G:\\template\\chip\\t\\text.png", 0);
	Mat dst = imread("G:\\template\\chip-picture\\samples\\01.png", 0);
	//模板信息生成中相关参数
	double lowThreshold = 60;
	double hightThreshold = 128;
	double min_score = 0.9;
	double greediness = 0.9;

	Point center;

	//检查是否成功加载图像
	if (src.empty()) {
		cout << "错误:没有加载模板图像" << endl;
		return -1;
	}
	if (dst.empty()) {
		cout << "错误:没有加载目标图像" << endl;
		return -1;
	}
	//显示模板图像
	namedWindow("src", WINDOW_AUTOSIZE);
	imshow("src", src);
	//显示目标图像
	namedWindow("dst", WINDOW_AUTOSIZE);
	imshow("dst", dst);
	//实例化
	ChipMatch chipmatch;

	//---------------高斯处理---------------//
	int kernelsize = 7;
	double sigma = 1.6;
	Mat src0 = chipmatch.GaussImg(src, kernelsize, sigma);
	/*namedWindow("yuan0", WINDOW_AUTOSIZE);
	imshow("yuan0", src0);*/
	//匹配信息结构体：定义为模板信息描述子中边缘点的相对坐标、X、Y方向的归一化梯度幅值
	struct Edge_Data {
		// 描述子中边缘点坐标集
		std::vector<Point*> edge_points;
		// 描述子中边缘点X偏导集
		std::vector<double*> edge_dxs;
		// 描述子中边缘点Y偏导集
		std::vector<double*> edge_dys;
	};
	//实例化边缘数据结构体
	Edge_Data edge_data;
	//初始化各指针
	Point* edge_point = nullptr;
	double* edge_dx = nullptr;
	double* edge_dy = nullptr;

	//-----------------------模板信息生成--------------------//

	


	//定义模板图像的最佳匹配点、相似度得分、偏移角度
	Point t;
	double match_score;
	double angle;

	//-----------------------搜索匹配计算--------------------//

	clock_t start_time = clock();
	match_score = chipmatch.MatchTemplate(dst, min_score, greediness, t, angle, &edge_point, &edge_dx, &edge_dy);
	clock_t end_time = clock();
	//获取搜索匹配过程的时间
	double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
	//输出匹配相似度、坐标点和匹配时间、偏移角度
	

	//-----------------------显示匹配结果----------------------//

	
	//释放定义的指针地址
	delete[] edge_point;
	delete[] edge_dx;
	delete[] edge_dy;

	waitKey(0);
	destroyAllWindows();

	return 0;
}