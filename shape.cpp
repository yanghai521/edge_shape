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
	// ��˹�˲�����ͼ��ƽ��ģ��
	//int kernelsize = 7;
	//double sigma = 1.6;
	GaussianBlur(src, gaussimg, Size(kernelsize, kernelsize), sigma);
	return gaussimg;
}

Mat ChipMatch::EdgeImage(Mat& guassimage, double low, double max, Point** t_Points, double** t_derivativeX, double** t_derivativeY) {
	Mat img = guassimage.clone();
	//�����ֵ����ģ��ͼ��Ŀ�������뱳��
	Mat binaryImage;
	threshold(img, binaryImage, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	//������Ĥ
	Mat dilatemask;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	dilate(binaryImage, dilatemask, kernel);
	//ģ��ͼ��ߡ���
	modelHeight = img.rows;
	modelWidth = img.cols;
	//��ȡģ��ͼ��Ŀ���Ե���x/y��������
	
    //***//
	
	//����ģ��ͼ��Ŀ���Ե���ݶȷ�ֵ������ָ����ͼ�����
	Mat df_grad1(modelHeight, modelWidth, CV_32FC1);
	Mat df_tan1(modelHeight, modelWidth, CV_64F);
	float left_p;
	float right_p;
	//�����ֵ���ƺ���ݶȷ�ֵ
	Mat G_n(modelHeight, modelWidth, CV_32F);
	Mat out_img;
	double tan0;
	double tan;
	//����ģ����ݶȷ�ֵͼ��ͽǶ�ͼ��
	for (int i = 1; i < modelHeight - 1; i++) {
		for (int j = 1; j < modelWidth - 1; j++) {
			float soble_x = imgX.at<float>(i, j);
			float soble_y = imgY.at<float>(i, j);
			df_grad1.at<float>(i, j) = sqrt(soble_x * soble_x + soble_y * soble_y);
			tan = atan2(soble_y, soble_x);
			df_tan1.at<double>(i, j) = tan;
		}
	}
	//���ݶȷ�ֵͼ���з�������ƣ�ϸ��ԭʼ��Ե
	








	//˫��ֵ����
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
	//�ɰ������ȥ��Ŀ��ͼ���Ե�ڵ��Ӳ�������
	Mat outputImage;
	out_img.copyTo(outputImage, dilatemask);

	//��ʼ��ģ����Ϣ�������е�����
	candidatePoint_n = 0;
	//ģ�������б�Ե���x��y���֮��
	int x_sum = 0;
	int y_sum = 0;
	int initial_MatchPoint = 0;
	//�����洢��Ե�����ꡢXƫ����Yƫ���Ķ��ڴ沢�������������ͷ�
	//�����������ٺ���֮�䴫��---�ı䴦
	



	//��ģ��δϡ��ǰ������ģ��Ĳο���
	


	//����ƽ������ο�������
	




	//-----------����Եϡ����ԡ�------------

	//1�����ȶ���Ӱ�����ƶȳ̶ȵı�׼��������ݶȷ�ֵ
	double dx;
	double dy;
	//��ǰ����ݶȷ�ֵ
	double M;
	//��ǰ��Ĺ�һ���ݶȷ�ֵ
	double gradientX;
	double gradientY;
	//ģ����Ϣ�������б�Ե�㼯��x��y����Ĺ�һ���ݶȷ�ֵ��
	Point point;
	vector<Point> point_all;
	vector<double> gradientXs;
	vector<double> gradientYs;
	//ѭ������
	int w = 0;
	//����r�Ĵ�С
	int r = 4;
	//�����Ե�ǰ��Ϊ���ĵ�r�����ڵ���������ݶȷ�ֵ
	double dx1, dy1, M1;
	//��������ݶȷ�ֵ������ݶȷ�ֵ��
	





	//2���ݶȷ�ֵ��ֵϡ��׶�
	int number = point_all.size();
	//ÿ�α����Ƚ��ҵ����������ݶȷ�ֵ��Ӧ�ĵ�����
	Point point0;
	//����ϡ������ѱ�Ե��Ĺ�һ���ݶȷ�ֵ��
	vector<double> gradientXs_best;
	vector<double> gradientYs_best;
	vector<Point> copy_point_all(point_all);
	//����ÿ�α����㼯�ҵ����������ݶȷ�ֵʱ����Ӧ�ڼ����еĴ���λ��
	double E_max;
	int maxIndex1;
	//����ϡ�����ѱ�Ե�㼯����Ե�������ݶȷ�ֵ��
	vector<double> E_best;
	vector<Point> point_best;
	//ѭ���ж���䣬ֱ����Ե�㼯�еĵ㾭����ݶȷ�ֵ�ȽϺ��ÿ�
	

		//��ÿ�α���Ѱַ������ݶȷ�ֵ�����r�����ڵ����������ꡢ��һ���ݶȷ�ֵ������ݶȷ�ֵ����0
		

		// If all points are empty, break the loop
		if (allEmpty) {
			break;
		}
	}
	Mat outimg(outputImage.size(), CV_8UC3);
	Mat templateimg0;
	outputImage.convertTo(templateimg0, CV_8UC3);
	cvtColor(templateimg0, outimg, COLOR_GRAY2BGR);
	//�������յ�ģ����Ϣ�����ӣ�������Ե����������ϵ��X��Y����Ĺ�һ���ݶȷ�ֵ
	
	return outimg;
}

double ChipMatch::MatchTemplate(Mat& searchImage, double min_score, double greediness, Point& resultPoint0, double& angle, Point** t_Points, double** t_derivativeX, double** t_derivativeY) {
	//��ز�����ʼ��
	double match_sum = 0;
	double m_score = 0;
	//Ŀ��ͼ���У���ĳһ������ʱ����Ӧģ����Ϣ�������еĵ�λ��
	int X;
	int Y;
	double template_GradientX, template_GradientY;
	double s_dx, s_dy;
	double r_score;
	int sum_m;
	Point match_point;
	//���ƣ���������ı�ԭʼͼ������
	Mat src = searchImage.clone();
	Mat src1 = searchImage.clone();

	//------------------��ȡ��С��Ӿ���--------------------
	
	//��ȡĿ��ͼ����Ŀ�����С��Ӿ���
	for (int i = 0; i < contour.size(); i++)
	{
		double area = contourArea(contour[i]);
		if (area > minArea)
		{
			MinBox = minAreaRect(contour[i]);
		}
	}
	//��С��Ӿ������ĵ���Ϊ������С������������ĵ㡢��һ�����������ĳ�ʼ��
	Point2f center = MinBox.center;
	//��ȡĿ��ͼ�������е��ˮƽ�ʹ�ֱ�����Ե
	
	//����Ŀ��ͼ���е���ݶȷ�ֵ��X��Y����Ĺ�һ���ݶȷ�ֵ
	double searchGradient;
	double search_GradientX;
	double search_GradientY;
	//����Ƕ�������Χ---����Ӻ���������Ĳ���
	int startAngle = -5;
	int endAngle = 5;
	double step = 0.1;
	//�������ڶ�ƫ�ƽǶ��£����ƶȵ÷ּ�������ƥ��㼯
	vector<double> r_scores;
	vector<Point> match_points;
	//����ڶ������������ĳ�ʼ��
	Point match_point0;
	//����ƥ�乫ʽ�����ƶ������ֵ���������е���ز���
	double M_min = 80;
	double score_min = 0.97;
	double score_a = 0.05;
	//����̰��ֹͣ��������
	double norm_min_score = min_score / candidatePoint_n;
	double norm_greediness = (1 - greediness * min_score) / (1 - min_score) / candidatePoint_n;//��1-greediness��Ϊ��1-min_score��

	//-----------------����ƥ�����--------------------

	//���ƶ������ֵ�������Ե�һ�����ж�ƫ�Ʒ���
	
		//��һ���Ƕ�������������Ҫ��ʼ�����ƶȵ÷֣�����ǰ��ͬ�Ƕȵ����ƶȵ÷ֵ�Ӱ��
		r_score = 0;
		//��һ����������������ѡȡ�Ƚ����ƶȣ������Ϊ��һ�����������ĳ�ʼ��
		
				//ͬ�ϣ���Բ�ͬ�������µ����ƶȵ÷־���ʼ��
				match_sum = 0;
				for (int m = 0; m < candidatePoint_n; m++) {
					//��ת�任��׼ģ����Ϣ�����ӣ�������ꡢ��һ���ݶȷ�ֵ
					
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
					//��Ŀ��ͼ����ĳһ������ʱ����Ӧģ����Ϣ�����ӵĵ��ݶȷ�ֵ���ڸ���ֵ���Ž������ƶȹ�ʽ����
					if (searchGradient >= M_min) {
						search_GradientX = s_dx / searchGradient;  //�ж�����Ϊ0������������λ��
						search_GradientY = s_dy / searchGradient;
						if ((template_GradientX != 0 || template_GradientY != 0) && (s_dx != 0 || s_dy != 0)) {
							match_sum += template_GradientX * search_GradientX + template_GradientY * search_GradientY;
						}
					}
					sum_m = m + 1;
					m_score = match_sum / sum_m;
					//̰�Ĺ�ʽ��ʱֹͣ�������
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
		//������һ�����������õ������ƶ����ֵ
		
		//�ڶ�������������������Χ�Ƚ�
		
				//����õ�Ķ��μ���
				
					//̰�Ĺ�ʽ��ʱֹͣ�������
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
		//�γ��ж�ƫ�Ʒ���ʱ����ͬ�Ƕ��µ����ƶȼ���ƥ��㼯
		r_scores.push_back(r_score);
		match_points.push_back(match_point);
	}
	//ȡ��0�Ƕ�һ�����ĽǶ�ģ��Ƚ����ƶȵ÷��жϷ���������ƫ��
	
		//��������ƫ�Ʒ����ʣ�½Ƕ�ģ����Ŀ��ͼ�����ƶ�
		
			//��һ����������������ѡȡ�Ƚ����ƶȣ������Ϊ��һ�����������ĳ�ʼ��
			
						if (searchGradient >= M_min) {
							search_GradientX = s_dx / searchGradient;  //�ж�����Ϊ0������������λ��
							search_GradientY = s_dy / searchGradient;
						
						//̰�Ĺ�ʽ��ʱֹͣ�������
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
			//�ڶ�������������������Χ�Ƚ�
			

						
						if (searchGradient >= M_min) {
							search_GradientX = s_dx / searchGradient;  //�ж�����Ϊ0������������λ��
							search_GradientY = s_dy / searchGradient;

						//̰�Ĺ�ʽ��ʱֹͣ�������
						
			//�γɸ�Ŀ��ͼ���£�����ƥ�������õ������ƶȼ�����ѱ�Ե�㼯
			r_scores.push_back(r_score);
			match_points.push_back(match_point);
			//�Ƚϳ�ʼ���ɽǶ�ģ�����ƶ���������ƶ���ֵ����С�ڳ�ʼ������ֵ������¸���ֵ
			if (s >= -1.1 && r_score <= score_min) {
				score_min = r_score - score_a;
			}
			//���ں��������Ƕ�ʱ������ĳ�Ƕ�ģ�����ƶ�С�ڸ��º�����ƶ���ֵ����ֹͣ������������
			else if (r_score <= score_min) {
				break;
			}
		}
	}
	//ȡ��0�Ƕ�һ�����ĽǶ�ģ��Ƚ����ƶȵ÷��жϷ���������ƫ��
	
		//��������ƫ�Ʒ����ʣ�½Ƕ�ģ����Ŀ��ͼ�����ƶ�
		
			//��һ����������������ѡȡ�Ƚ����ƶȣ������Ϊ��һ�����������ĳ�ʼ��
			
						//̰�Ĺ�ʽ��ʱֹͣ�������
						
			//�ڶ�������������������Χ�Ƚ�
			
						//̰�Ĺ�ʽ��ʱֹͣ�������
						
			//�γɸ�Ŀ��ͼ���£�����ƥ�������õ������ƶȼ�����ѱ�Ե�㼯
			
			//�Ƚϳ�ʼ���ɽǶ�ģ�����ƶ���������ƶ���ֵ����С�ڳ�ʼ������ֵ������¸���ֵ
			
			//���ں��������Ƕ�ʱ������ĳ�Ƕ�ģ�����ƶ�С�ڸ��º�����ƶ���ֵ����ֹͣ������������
			

	//-------------Ѱ�����ƥ����ƫ�ƽǶ�ʱ��ģ��----------------

	double maxIndex;
	//��λ���ƶȼ���������ƶ����ڵ�λ��
	auto score_max = std::max_element(r_scores.begin(), r_scores.end());
	if (score_max != r_scores.end()) {
		maxIndex = std::distance(r_scores.begin(), score_max);
	}
	//Ѱ�����÷�ֵ��Ӧ��ƥ��㡢���ƶȵ÷�
	resultPoint0.x = match_points[maxIndex].x;
	resultPoint0.y = match_points[maxIndex].y;
	double maxScore = *score_max;
	//Ѱ��������ƶȶ�Ӧ��ƫ�ƽǶȣ����ڱȽ�ƫ�Ʒ���ʱ�ĽǶ�����
	
	//��ƫ�ƽǶ��ڸ���Ƕ�����
	
	//��ƫ�Ʒ���������Ƕ�����
	
}

ChipMatch::~ChipMatch(void)
{
	//������ʹ�����ʱ���ǵ�delete���ͷ��ڴ�
	//new����Ҫ�����ͷţ�delete���ڽ��й��̱��ʱ������ʹ�ö��ڴ棬ջ�ڴ�ֻ�Ƕ��ݵı��� 
	delete[] Points;
	delete[] derivativeX;
	delete[] derivativeY;


}

//������ʱ�����ڴ�
void ChipMatch::CreateMatrix(double**& matrix, int, int)
{
	matrix = new double* [(size_t)modelHeight - 2];
	for (int i = 0; i < (modelHeight - 2); i++) {
		matrix[i] = new double[(size_t)modelWidth - 2];
	}
}
//�ͷ���ʱ�����ڴ�
void ChipMatch::ReleaseMatrix(double**& matrix, int size)
{
	for (int i = 0; i < modelHeight; i++) {
		delete[] matrix[i];
	}
}
