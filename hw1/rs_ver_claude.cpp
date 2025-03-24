#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

template <typename T>
Mat myrotate(const Mat input, float angle, const char* opt);

int main()
{
	Mat input, rotated;
	
	// Read each image
	input = imread("lena.jpg");

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
					// Nearest neighbor interpolation
					int nx = round(x);
					int ny = round(y);
					
					// 경계 검사
					nx = max(0, min(nx, col - 1));
					ny = max(0, min(ny, row - 1));
					
					output.at<T>(i, j) = input.at<T>(ny, nx);
				}
				else if (!strcmp(opt, "bilinear")) {
					// Bilinear interpolation
					int x1 = floor(x);
					int y1 = floor(y);
					int x2 = ceil(x);
					int y2 = ceil(y);
					
					// 경계 검사
					x1 = max(0, min(x1, col - 1));
					y1 = max(0, min(y1, row - 1));
					x2 = max(0, min(x2, col - 1));
					y2 = max(0, min(y2, row - 1));
					
					// 보간 가중치 계산
					float dx = x - x1;
					float dy = y - y1;
					
					// 가중치를 적용한 픽셀 값 계산
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