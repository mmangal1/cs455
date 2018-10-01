#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;


Mat add_sub_img(Mat origImg, Mat newImg){
	for (int x = 0; x < newImg.rows; x++){
		for(int y = 0; y < newImg.cols; y++){
			newImg.at<uchar>(x,y) = saturate_cast<uchar>(origImg.at<uchar>(x,y) + (origImg.at<uchar>(x,y) - newImg.at<uchar>(x,y)));
		}
	}
	return newImg;
	
}

Mat unsharp_mask(Mat img){
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
		for(int y = 1; y < new_img.rows-1; y++){
			new_img.at<uchar>(x, y) = GausianMask[0][0] * img.at<uchar>(x-1,y-1) + GausianMask[0][1] * img.at<uchar>(x-1, y) + GausianMask[0][2] * img.at<uchar>(x-1, y+1)
						+ GausianMask[1][0] * img.at<uchar>(x, y-1)  + GausianMask[1][1] * img.at<uchar>(x, y)   + GausianMask[1][2] * img.at<uchar>(x, y+1)
						+ GausianMask[2][0] * img.at<uchar>(x+1,y-1) + GausianMask[2][1] * img.at<uchar>(x+1, y) + GausianMask[2][2] * img.at<uchar>(x+1, y+1);
		}
	}

	namedWindow("Blurred Ant Image", WINDOW_AUTOSIZE);
	imshow("Blurred Ant Image", new_img);

	new_img = add_sub_img(img, new_img);	

	return new_img;
}


int main(){
	string ant_image = "../img/ant_gray.bmp";
	string statue_image = "../img/basel_gray.bmp";
	
	//-------------- Original Ant Image ---------------
	Mat ant_img = imread(ant_image, IMREAD_GRAYSCALE);
	namedWindow("Original Ant Image", WINDOW_AUTOSIZE);
	imshow("Original Ant Image", ant_img);
	
	//-------------- Enhanced Ant Image ---------------
	Mat enhanced_ant_img = unsharp_mask(ant_img);
	namedWindow("Enhanced Ant Image", WINDOW_AUTOSIZE);
	imshow("Enhanced Ant Image", enhanced_ant_img);
	
	//-------------- Original Basel Image ---------------
	Mat statue_img = imread(statue_image, IMREAD_GRAYSCALE);
	namedWindow("Original Statue Image", WINDOW_AUTOSIZE);
	imshow("Original Statue Image", statue_img);
	
	//-------------- Enhanced Ant Image ---------------
	Mat enhanced_statue_img = unsharp_mask(statue_img);
	namedWindow("Enhanced Statue Image", WINDOW_AUTOSIZE);
	imshow("Enhanced Statue Image", enhanced_statue_img);

	waitKey();
	destroyAllWindows();	

}
