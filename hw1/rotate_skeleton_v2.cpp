#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

template <typename T>
Mat myrotate(const Mat input, float angle, const char* opt);

int main()
{
	Mat input, rotated;
	
	// Read each image
	input = imread("/Users/rainyforest/Desktop/2025-1_OSP-hw/hw1/lena.jpg");


	// Check for invalid input
	if (!input.data) {
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	
	// original image
	namedWindow("image");
	imshow("image", input);

	rotated = myrotate<Vec3b>(input, 45, "bilinear");

	// rotated image
	namedWindow("rotated");
	imshow("rotated", rotated);

	waitKey(0);

	return 0;
}

template <typename T>
Mat myrotate(const Mat input, float angle, const char* opt) {
	int row = input.rows;
	int col = input.cols;

	float radian = angle * CV_PI / 180;

	float sq_row = ceil(row * sin(radian) + col * cos(radian));
	float sq_col = ceil(col * sin(radian) + row * cos(radian));

	Mat output = Mat::zeros(sq_row, sq_col, input.type());

	for (int i = 0; i < sq_row; i++) {
		for (int j = 0; j < sq_col; j++) {
			float x = (j - sq_col / 2) * cos(radian) - (i - sq_row / 2) * sin(radian) + col / 2;
			float y = (j - sq_col / 2) * sin(radian) + (i - sq_row / 2) * cos(radian) + row / 2;

			if ((y >= 0) && (y <= (row - 1)) && (x >= 0) && (x <= (col - 1))) {
				if (!strcmp(opt, "nearest")) {

					// NN interpolation
					// round() : 가장 가까운 픽셀로 위치를 정수화함
					int nnx = round(x);
					int nny = round(y);

					/* 여기서 (x,y)는 input coordinate
					(i,j)는  output coordinate */

					/* OpenCV는 .at<T>(**행**, **열**) = .at<T>(y, x) 순서 유의 ! */

					output.at<T>(i, j) = input.at<T>(nny, nnx);




				}
				else if (!strcmp(opt, "bilinear")) {

					// Bilinear interpolation
					/* float 좌표(x,y)를 감싸는 4개의 정수 좌표를 찾고, 가중치*평균하여 구하기 */

					int x1 = floor(x);
					int y1 = floor(y);
					int x2 = ceil(x);
					int y2 = ceil(y);

					/* floor(x) : x보다 작거나 같은 가장 큰 정수 (왼쪽 끝 값의 좌표) 
					ceil(x) : x보다 크거나 같은 가장 작은 정수 (오른쪽 끝 값의 좌표) */

					float dx = x-x1;
					float dy = y-y1;


					output.at<T>(i, j) = (1 - dx) * (1 - dy) * input.at<T>(y1, x1) +
										 dx * (1 - dy) * input.at<T>(y1, x2) +
										 (1 - dx) * dy * input.at<T>(y2, x1) +
										 dx * dy * input.at<T>(y2, x2);
				}
			}
		}
	}

	return output;
}