#include "shape.h"
ChipMatch::ChipMatch(void)
{
	modelWidth = 0;
	modelHeight = 0;
	candidatePoint_n = 0;
}

Mat ChipMatch::GaussImg(Mat& srcimage, int kernelsize, double sigma)
{
	Mat src = srcimage.clone();
	Mat gaussimg;
	// 高斯滤波器对图像平滑模糊
	//int kernelsize = 7;
	//double sigma = 1.6;
	GaussianBlur(src, gaussimg, Size(kernelsize, kernelsize), sigma);
	return gaussimg;
}

Mat ChipMatch::EdgeImage(Mat& guassimage, double low, double max, Point** t_Points, double** t_derivativeX, double** t_derivativeY) {
	Mat img = guassimage.clone();
	//大津阈值区分模板图像目标区域与背景
	Mat binaryImage;
	threshold(img, binaryImage, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	//制作掩膜
	Mat dilatemask;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	dilate(binaryImage, dilatemask, kernel);
	//模板图像高、宽
	modelHeight = img.rows;
	modelWidth = img.cols;
	//提取模板图像目标边缘点的x/y方向向量
	
    //***//
	
	//创建模板图像目标边缘点梯度幅值、向量指向方向图像矩阵
	Mat df_grad1(modelHeight, modelWidth, CV_32FC1);
	Mat df_tan1(modelHeight, modelWidth, CV_64F);
	float left_p;
	float right_p;
	//非最大值抑制后的梯度幅值
	Mat G_n(modelHeight, modelWidth, CV_32F);
	Mat out_img;
	double tan0;
	double tan;
	//计算模板的梯度幅值图像和角度图像
	for (int i = 1; i < modelHeight - 1; i++) {
		for (int j = 1; j < modelWidth - 1; j++) {
			float soble_x = imgX.at<float>(i, j);
			float soble_y = imgY.at<float>(i, j);
			df_grad1.at<float>(i, j) = sqrt(soble_x * soble_x + soble_y * soble_y);
			tan = atan2(soble_y, soble_x);
			df_tan1.at<double>(i, j) = tan;
		}
	}
	//在梯度幅值图像中非最大抑制：细化原始边缘
	








	//双阈值处理
	out_img = G_n.clone();
	for (int i = 1; i < modelHeight - 1; i++) {
		for (int j = 1; j < modelWidth - 1; j++) {
			if (out_img.at<float>(i, j) < low) {
				out_img.at<float>(i, j) = 0;
			}
			else if (out_img.at<float>(i - 1, j - 1) < max &&
				out_img.at<float>(i, j - 1) < max &&
				out_img.at<float>(i + 1, j - 1) < max &&
				out_img.at<float>(i - 1, j) < max &&
				out_img.at<float>(i + 1, j) < max &&
				out_img.at<float>(i - 1, j + 1) < max &&
				out_img.at<float>(i, j + 1) < max &&
				out_img.at<float>(i + 1, j + 1) < max) {
				out_img.at<float>(i, j) = 0;
			}
		}
	}
	Mat out_img0;
	out_img.convertTo(out_img0, CV_8U);
	//蒙版操作：去除目标图像边缘内的杂波和噪声
	Mat outputImage;
	out_img.copyTo(outputImage, dilatemask);

	//初始化模板信息描述子中点数量
	candidatePoint_n = 0;
	//模板中所有边缘点的x、y相加之和
	int x_sum = 0;
	int y_sum = 0;
	int initial_MatchPoint = 0;
	//创建存储边缘点坐标、X偏导、Y偏导的堆内存并在析构函数中释放
	//公共化、不再函数之间传递---改变处
	



	//在模板未稀疏前，计算模板的参考点
	


	//算术平均计算参考点坐标
	




	//-----------“边缘稀疏策略”------------

	//1、首先定义影响相似度程度的标准——相对梯度幅值
	double dx;
	double dy;
	//当前点的梯度幅值
	double M;
	//当前点的归一化梯度幅值
	double gradientX;
	double gradientY;
	//模板信息描述子中边缘点集、x、y方向的归一化梯度幅值集
	Point point;
	vector<Point> point_all;
	vector<double> gradientXs;
	vector<double> gradientYs;
	//循环计数
	int w = 0;
	//邻域r的大小
	int r = 4;
	//定义以当前点为中心的r邻域内的其他点的梯度幅值
	double dx1, dy1, M1;
	//定义相对梯度幅值、相对梯度幅值集
	





	//2、梯度幅值阈值稀疏阶段
	int number = point_all.size();
	//每次遍历比较找到的最大相对梯度幅值对应的点坐标
	Point point0;
	//定义稀疏后的最佳边缘点的归一化梯度幅值集
	vector<double> gradientXs_best;
	vector<double> gradientYs_best;
	vector<Point> copy_point_all(point_all);
	//定义每次遍历点集找到的最大相对梯度幅值时，对应在集合中的次序位置
	double E_max;
	int maxIndex1;
	//定义稀疏后最佳边缘点集、边缘点的相对梯度幅值集
	vector<double> E_best;
	vector<Point> point_best;
	//循环判断语句，直至边缘点集中的点经相对梯度幅值比较后置空
	

		//对每次遍历寻址的相对梯度幅值坐标点r邻域内的其他点坐标、归一化梯度幅值和相对梯度幅值均置0
		

		// If all points are empty, break the loop
		if (allEmpty) {
			break;
		}
	}
	Mat outimg(outputImage.size(), CV_8UC3);
	Mat templateimg0;
	outputImage.convertTo(templateimg0, CV_8UC3);
	cvtColor(templateimg0, outimg, COLOR_GRAY2BGR);
	//生成最终的模板信息描述子，包括边缘点相对坐标关系、X和Y方向的归一化梯度幅值
	
	return outimg;
}

double ChipMatch::MatchTemplate(Mat& searchImage, double min_score, double greediness, Point& resultPoint0, double& angle, Point** t_Points, double** t_derivativeX, double** t_derivativeY) {
	//相关参数初始化
	double match_sum = 0;
	double m_score = 0;
	//目标图像中，在某一搜索点时，对应模板信息描述子中的点位置
	int X;
	int Y;
	double template_GradientX, template_GradientY;
	double s_dx, s_dy;
	double r_score;
	int sum_m;
	Point match_point;
	//复制，不会意外改变原始图像数据
	Mat src = searchImage.clone();
	Mat src1 = searchImage.clone();

	//------------------获取最小外接矩形--------------------
	
	//提取目标图像中目标的最小外接矩形
	for (int i = 0; i < contour.size(); i++)
	{
		double area = contourArea(contour[i]);
		if (area > minArea)
		{
			MinBox = minAreaRect(contour[i]);
		}
	}
	//最小外接矩形中心点作为建立最小搜索区域的中心点、第一步邻域搜索的初始点
	Point2f center = MinBox.center;
	//获取目标图像中所有点的水平和垂直方向边缘
	
	//定义目标图像中点的梯度幅值、X和Y方向的归一化梯度幅值
	double searchGradient;
	double search_GradientX;
	double search_GradientY;
	//定义角度搜索范围---改外接函数可输入的参数
	int startAngle = -5;
	int endAngle = 5;
	double step = 0.1;
	//定义在众多偏移角度下，相似度得分集和最终匹配点集
	vector<double> r_scores;
	vector<Point> match_points;
	//定义第二步邻域搜索的初始点
	Point match_point0;
	//相似匹配公式、相似度最低阈值搜索策略中的相关参数
	double M_min = 80;
	double score_min = 0.97;
	double score_a = 0.05;
	//计算贪心停止搜索参数
	double norm_min_score = min_score / candidatePoint_n;
	double norm_greediness = (1 - greediness * min_score) / (1 - min_score) / candidatePoint_n;//将1-greediness改为了1-min_score；

	//-----------------搜索匹配过程--------------------

	//相似度最低阈值搜索策略第一步：判断偏移方向
	
		//在一个角度搜索结束后，需要初始化相似度得分，避免前后不同角度的相似度得分的影响
		r_score = 0;
		//第一步邻域搜索，隔点选取比较相似度，最大者为下一步邻域搜索的初始点
		
				//同上，需对不同搜索点下的相似度得分均初始化
				match_sum = 0;
				for (int m = 0; m < candidatePoint_n; m++) {
					//旋转变换基准模板信息描述子，相对坐标、归一化梯度幅值
					
					X = i + a;
					Y = j + b;
					template_GradientX = c;
					template_GradientY = d;
					if (X<0 || Y<0 || X>searchHeight - 1 || Y>searchWidth - 1) {
						continue;
					}
					s_dx = static_cast<double>(s_x.at<float>(X, Y));
					s_dy = static_cast<double>(s_y.at<float>(X, Y));
					searchGradient = sqrt(s_dx * s_dx + s_dy * s_dy);
					//当目标图像在某一搜索点时，对应模板信息描述子的点梯度幅值大于该阈值，才进行相似度公式计算
					if (searchGradient >= M_min) {
						search_GradientX = s_dx / searchGradient;  //判断设置为0界限条件放置位置
						search_GradientY = s_dy / searchGradient;
						if ((template_GradientX != 0 || template_GradientY != 0) && (s_dx != 0 || s_dy != 0)) {
							match_sum += template_GradientX * search_GradientX + template_GradientY * search_GradientY;
						}
					}
					sum_m = m + 1;
					m_score = match_sum / sum_m;
					//贪心公式及时停止多余计算
					if (m_score < MIN((min_score - 1) + norm_greediness * (sum_m), norm_min_score * (sum_m))) {
						break;
					}
				}
				if (m_score > r_score) {
					r_score = m_score;
					match_point0.x = i;
					match_point0.y = j;
				}
			}
		}
		//保留第一步邻域搜索得到的相似度最大值
		
		//第二步邻域搜索：八邻域范围比较
		
				//避免该点的二次计算
				
					//贪心公式及时停止多余计算
					if (m_score < MIN((min_score - 1) + norm_greediness * (sum_m), norm_min_score * (sum_m))) {
						break;
					}
				}
				if (m_score > r_score) {
					r_score = m_score;
					match_point.x = i;
					match_point.y = j;
				}
			}
		}
		//形成判断偏移方向时，不同角度下的相似度集和匹配点集
		r_scores.push_back(r_score);
		match_points.push_back(match_point);
	}
	//取距0角度一定间距的角度模板比较相似度得分判断方向，若负向偏移
	
		//搜索负向偏移方向的剩下角度模板与目标图像相似度
		
			//第一步邻域搜索，隔点选取比较相似度，最大者为下一步邻域搜索的初始点
			
						if (searchGradient >= M_min) {
							search_GradientX = s_dx / searchGradient;  //判断设置为0界限条件放置位置
							search_GradientY = s_dy / searchGradient;
						
						//贪心公式及时停止多余计算
						if (m_score < MIN((min_score - 1) + norm_greediness * (sum_m), norm_min_score * (sum_m))) {
							break;
						}
					}
					if (m_score > r_score) {
						r_score = m_score;
						match_point0.x = i;
						match_point0.y = j;
					}
				}
			}
			double copy_score = r_score;
			r_score = 0;
			//第二步邻域搜索：八邻域范围比较
			

						
						if (searchGradient >= M_min) {
							search_GradientX = s_dx / searchGradient;  //判断设置为0界限条件放置位置
							search_GradientY = s_dy / searchGradient;

						//贪心公式及时停止多余计算
						
			//形成该目标图像下，最终匹配搜索得到的相似度集、最佳边缘点集
			r_scores.push_back(r_score);
			match_points.push_back(match_point);
			//比较初始若干角度模板相似度与最低相似度阈值，若小于初始设置阈值，则更新该阈值
			if (s >= -1.1 && r_score <= score_min) {
				score_min = r_score - score_a;
			}
			//若在后续搜索角度时，出现某角度模板相似度小于更新后的相似度阈值，则停止接下来的搜索
			else if (r_score <= score_min) {
				break;
			}
		}
	}
	//取距0角度一定间距的角度模板比较相似度得分判断方向，若正向偏移
	
		//搜索正向偏移方向的剩下角度模板与目标图像相似度
		
			//第一步邻域搜索，隔点选取比较相似度，最大者为下一步邻域搜索的初始点
			
						//贪心公式及时停止多余计算
						
			//第二步邻域搜索：八邻域范围比较
			
						//贪心公式及时停止多余计算
						
			//形成该目标图像下，最终匹配搜索得到的相似度集、最佳边缘点集
			
			//比较初始若干角度模板相似度与最低相似度阈值，若小于初始设置阈值，则更新该阈值
			
			//若在后续搜索角度时，出现某角度模板相似度小于更新后的相似度阈值，则停止接下来的搜索
			

	//-------------寻找最佳匹配点和偏移角度时的模板----------------

	double maxIndex;
	//定位相似度集中最大相似度所在的位置
	auto score_max = std::max_element(r_scores.begin(), r_scores.end());
	if (score_max != r_scores.end()) {
		maxIndex = std::distance(r_scores.begin(), score_max);
	}
	//寻找最大得分值对应的匹配点、相似度得分
	resultPoint0.x = match_points[maxIndex].x;
	resultPoint0.y = match_points[maxIndex].y;
	double maxScore = *score_max;
	//寻找最大相似度对应的偏移角度，若在比较偏移方向时的角度区段
	
	//若偏移角度在负向角度区段
	
	//若偏移方向在正向角度区段
	
}

ChipMatch::~ChipMatch(void)
{
	//当对象使用完毕时，记得delete，释放内存
	//new后面要进行释放，delete；在进行工程编程时，建议使用堆内存，栈内存只是短暂的变量 
	delete[] Points;
	delete[] derivativeX;
	delete[] derivativeY;


}

//分配临时矩阵内存
void ChipMatch::CreateMatrix(double**& matrix, int, int)
{
	matrix = new double* [(size_t)modelHeight - 2];
	for (int i = 0; i < (modelHeight - 2); i++) {
		matrix[i] = new double[(size_t)modelWidth - 2];
	}
}
//释放临时矩阵内存
void ChipMatch::ReleaseMatrix(double**& matrix, int size)
{
	for (int i = 0; i < modelHeight; i++) {
		delete[] matrix[i];
	}
}
