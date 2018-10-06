#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

int calculateXGradient(Mat img, int x, int y){
	double XGrad[3][3] = {
		{-1, -2, -1},
		{0, 0, 0},
		{1, 2, 1}
	};
	
	int gradient = XGrad[0][0] * img.at<uchar>(x-1,y-1) + XGrad[0][1] * img.at<uchar>(x-1, y) + XGrad[0][2] * img.at<uchar>(x-1, y+1)
		     + XGrad[1][0] * img.at<uchar>(x, y-1)  + XGrad[1][1] * img.at<uchar>(x, y)   + XGrad[1][2] * img.at<uchar>(x, y+1)
		     + XGrad[2][0] * img.at<uchar>(x+1,y-1) + XGrad[2][1] * img.at<uchar>(x+1, y) + XGrad[2][2] * img.at<uchar>(x+1, y+1);

	return gradient;
}

int calculateYGradient(Mat img, int x, int y){
	double YGrad[3][3] = {
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1}
	};
	
	int gradient = YGrad[0][0] * img.at<uchar>(x-1,y-1) + YGrad[0][1] * img.at<uchar>(x-1, y) + YGrad[0][2] * img.at<uchar>(x-1, y+1)
		     + YGrad[1][0] * img.at<uchar>(x, y-1)  + YGrad[1][1] * img.at<uchar>(x, y)   + YGrad[1][2] * img.at<uchar>(x, y+1)
		     + YGrad[2][0] * img.at<uchar>(x+1,y-1) + YGrad[2][1] * img.at<uchar>(x+1, y) + YGrad[2][2] * img.at<uchar>(x+1, y+1);

	return gradient;
}

Mat add_sub_img(Mat origImg, Mat newImg){
	for (int x = 0; x < newImg.rows; x++){
		for(int y = 0; y < newImg.cols; y++){
			newImg.at<uchar>(x,y) = saturate_cast<uchar>(origImg.at<uchar>(x,y) + (origImg.at<uchar>(x,y) - newImg.at<uchar>(x,y)));
		}
	}
	return newImg;
	
}

Mat unsharp_mask(Mat img, string imname){
	Mat new_img = img.clone();
	
	for(int x = 0; x < img.rows; x++){
		for(int y = 0; y < img.cols; y++){
			new_img.at<uchar>(x,y) = 0; 
		}
	}

	//Gaussian filter for blurring
	double GausianMask[3][3] = {
		{1/16.0, 2/16.0, 1/16.0},
		{2/16.0, 4/16.0, 2/16.0},
		{1/16.0, 2/16.0, 1/16.0}
	};

	for(int x = 1; x < new_img.rows-1; x++){
		for(int y = 1; y < new_img.cols-1; y++){
			new_img.at<uchar>(x, y) = GausianMask[0][0] * img.at<uchar>(x-1,y-1) + GausianMask[0][1] * img.at<uchar>(x-1, y) + GausianMask[0][2] * img.at<uchar>(x-1, y+1)
						+ GausianMask[1][0] * img.at<uchar>(x, y-1)  + GausianMask[1][1] * img.at<uchar>(x, y)   + GausianMask[1][2] * img.at<uchar>(x, y+1)
						+ GausianMask[2][0] * img.at<uchar>(x+1,y-1) + GausianMask[2][1] * img.at<uchar>(x+1, y) + GausianMask[2][2] * img.at<uchar>(x+1, y+1);
		}
	}

	string name = "Blurred " + imname + " Image";
	namedWindow(name, WINDOW_AUTOSIZE);
	imshow(name, new_img);

	new_img = add_sub_img(img, new_img);	

	return new_img;
}

Mat sobel(Mat img){

	Mat new_img = img.clone();
	
	for(int x = 0; x < img.rows; x++){
		for(int y = 0; y < img.cols; y++){
			new_img.at<uchar>(x,y) = 0; 
		}
	}

	for(int x = 0; x < img.rows; x++){
		for(int y = 0; y < img.cols; y++){
			int horizontal_grad = calculateXGradient(img, x, y);
			int vertical_grad = calculateYGradient(img, x, y);
			int total_grad = abs(horizontal_grad) + abs(vertical_grad);
	
			if(total_grad > 255){
				total_grad = 255;
			}else if(total_grad < 0){
				total_grad = 0;
			}

			new_img.at<uchar>(x, y) = total_grad;
		}
	}

	return new_img;
}


Mat LOGMask(int dim, double sigma, Mat image){
	const double PI = 3.1415;
	
	int MASK[dim][dim];
	double sigma2 = sigma * sigma;

	double left = pow( 1/(sqrt(2*PI)*sigma2), 2);

	int max = dim/2;
	int min = -(dim/2);	
	int x_count = 0;

	for(int y = -dim/2; y <= dim/2; y++){
		int y_count = 0;
		for(int x = -dim/2; x <= dim/2; x++){
			double mid = ((x*x + y*y)/sigma2);
			double right = exp(mid * -1/2);
			double val = left * (mid - 2) * right;
			if(sigma == 5.0){
				if((y+max) >= 3 && (y+max) <= 7 && (x+max) >= 2 && (x+max) <= 8){
					val = val;
				}else{
					val = val * -1;
				}
				MASK[x_count][y_count] = val * 50000;
			}else{
				MASK[x_count][y_count] = val * 500;	
			}

			y_count++;
		}
		x_count++;
	}

	for(int x = 0; x < dim; x++){
		for(int y = 0; y < dim; y++){
			cout << MASK[x][y] << '\t';
		}
		cout << endl;
	}
	cout << endl << endl;

	Mat fin_img = image.clone();
	for(int x = 0; x < image.rows; x++){
		for(int y = 0; y < image.cols; y++){
			fin_img.at<uchar>(x,y) = 0; 
		}
	}
	
	for(int x = 0; x < fin_img.rows; x++){
		for(int y = 0; y < fin_img.cols; y++){
			double sum = 0;
			for(int i = min; i <= max; i++){
				for(int j = min; j <= max; j++){
					sum += MASK[i+max][j+max] * (double)image.at<uchar>(x - i, y - j);
				}
			}
			fin_img.at<uchar>(x,y) = saturate_cast<uchar>(sum);
		}
	}

	return fin_img;
}

int main(){

	string ant_image = "../img/ant_gray.bmp";
	string statue_image = "../img/basel_gray.bmp";
	
	//-------------- Original Ant Image ---------------
	Mat ant_img = imread(ant_image, IMREAD_GRAYSCALE);
	namedWindow("Original Ant Image", WINDOW_AUTOSIZE);
	imshow("Original Ant Image", ant_img);
	
	//-------------- Enhanced Ant Image ---------------
	Mat enhanced_ant_img = unsharp_mask(ant_img, "Ant");
	namedWindow("Enhanced Ant Image", WINDOW_AUTOSIZE);
	imshow("Enhanced Ant Image", enhanced_ant_img);
	
	//-------------- Sobel Ant Image ------------------
	Mat sobel_ant_img = sobel(ant_img);
	namedWindow("Sobel Ant Image", WINDOW_AUTOSIZE);
	imshow("Sobel Ant Image", sobel_ant_img);
	
	//-------------- LOG 1.4 Ant Image ----------------------
	Mat log_ant_img = LOGMask(7, 1.4, ant_img);
	namedWindow("E1_1", WINDOW_AUTOSIZE);
	imshow("E1_1", log_ant_img);
	
	//-------------- LOG 5.0 Ant Image ----------------------
	log_ant_img = LOGMask(11, 5.0, ant_img);
	namedWindow("E1_2", WINDOW_AUTOSIZE);
	imshow("E1_2", log_ant_img);
	
	//-------------- Original Basel Image ---------------
	Mat statue_img = imread(statue_image, IMREAD_GRAYSCALE);
	namedWindow("Original Statue Image", WINDOW_AUTOSIZE);
	imshow("Original Statue Image", statue_img);
	
	//-------------- Enhanced Basel Image ---------------
	Mat enhanced_statue_img = unsharp_mask(statue_img, "Statue");
	namedWindow("Enhanced Statue Image", WINDOW_AUTOSIZE);
	imshow("Enhanced Statue Image", enhanced_statue_img);

	//-------------- Sobel Basel Image ------------------
	Mat sobel_statue_img = sobel(statue_img);
	namedWindow("Sobel Statue Image", WINDOW_AUTOSIZE);
	imshow("Sobel Statue Image", sobel_statue_img);

	//-------------- LOG 1.4 Basel Image ----------------------
	Mat log_basel_img = LOGMask(7, 1.4, statue_img);
	namedWindow("E2_1", WINDOW_AUTOSIZE);
	imshow("E2_1", log_basel_img);
	
	//-------------- LOG 5.0 Ant Image ----------------------
	log_basel_img = LOGMask(11, 5.0, statue_img);
	namedWindow("E2_2", WINDOW_AUTOSIZE);
	imshow("E2_2", log_basel_img);

	waitKey();
	destroyAllWindows();	

}
