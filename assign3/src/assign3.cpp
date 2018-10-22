#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

#define PI 3.1415

Mat DCTImage(Mat orig_image){
	int h = orig_image.rows - (orig_image.rows % 8);
	int w = orig_image.cols - (orig_image.cols % 8);

	Mat fin_img = orig_image.clone();
	for(int x = 0; x < fin_img.rows; x++){
		for(int y = 0; y < fin_img.cols; y++){
			fin_img.at<Vec3b>(x, y)[0] = 0;
			fin_img.at<Vec3b>(x, y)[1] = 0;
			fin_img.at<Vec3b>(x, y)[2] = 0;
		}
	}
	
	double alpha_i = 1.0;
	double alpha_j = 1.0;
	
	for(int x = 0; x < h; x+= 8){
		for(int y = 0; y < w; y+= 8){
			for(int i = x; i < x+8; i++){
				for(int j = y; j < y+8; j++){
					float t = 0.0;
					if(i == 0)
						alpha_i = 1.0/sqrt(2.0);
					else
						alpha_i = 1.0;
					if(j == 0)
						alpha_j = 1.0/sqrt(2.0);
					else
						alpha_j = 1.0;

					for(int p = x; p < x+8; p++){
						for(int q = y; q < y+8; q++){
							t += (orig_image.at<Vec3b>(p,q)[2]) * (cos((((2*p)+1)) * ((i* M_PI))/ (2*8))) * (cos((((2*q) + 1)) * ((j * M_PI)) / (2*8)));	
						}
					}
					t *= ((1/sqrt(2*8))) * (alpha_i * alpha_j);
					fin_img.at<Vec3b>(i, j)[2] = int(t);
				}
			}
		}
	}
	
	return fin_img;
}

Mat IDCTImage(Mat orig_image){
	int h = orig_image.rows - (orig_image.rows % 8);
	int w = orig_image.cols - (orig_image.cols % 8);

	Mat fin_img = orig_image.clone();
	for(int x = 0; x < fin_img.rows; x++){
		for(int y = 0; y < fin_img.cols; y++){
			fin_img.at<Vec3b>(x, y)[0] = 0;
			fin_img.at<Vec3b>(x, y)[1] = 0;
			fin_img.at<Vec3b>(x, y)[2] = 0;
		}
	}

	float t;
	double alpha_i = 1.0;
	double alpha_j = 1.0;
	
	for(int x = 0; x < h; x+= 8){
		for(int y = 0; y < w; y+= 8){
			for(int i = x; i < x+8; i++){
				for(int j = y; j < y+8; j++){
					t = 0.0;
					if(i == 0)
						alpha_i = 1.0/sqrt(2.0);
					else
						alpha_i = 1.0;
					if(j == 0)
						alpha_j = 1.0/sqrt(2.0);
					else
						alpha_j = 1.0;

					for(int p = x; p < x+8; p++){
						for(int q = y; q < y+8; q++){
							t = t + (orig_image.at<Vec3b>(p,q)[2])  * (cos((((2 * i) + 1)) * ((p * M_PI)) / (2 * 8))) * (cos((((2 * j) + 1)) * ((q * M_PI)) / (2 * 8))) * (alpha_i*alpha_j);
						}
					}
					t *= ((2/sqrt(8*8)));
					fin_img.at<Vec3b>(i, j)[2] = int(t);
				}
			}
		}
	}
	return fin_img;
}


Mat DCTImage_DC(Mat orig_image){
	int h = orig_image.rows - (orig_image.rows % 8);
	int w = orig_image.cols - (orig_image.cols % 8);

	Mat fin_img = orig_image.clone();
	
	for(int x = 0; x < h; x+= 8){
		for(int y = 0; y < w; y+= 8){
			for(int i = x; i < x+8; i++){
				for(int j = y; j < y+8; j++){
					if(i == x && j == y){
						continue;
					}else{
						fin_img.at<Vec3b>(i,j)[2] = 0;
					}
				}
			}
		}
	}
	
	return fin_img;
}

Mat DCTImage_9(Mat orig_image){
	int h = orig_image.rows - (orig_image.rows % 8);
	int w = orig_image.cols - (orig_image.cols % 8);

	Mat fin_img = orig_image.clone();
	
	for(int x = 0; x < h; x+= 8){
		for(int y = 0; y < w; y+= 8){
			for(int i = x; i < x+8; i++){
				for(int j = y; j < y+8; j++){
					for(int p = x; p < x+8; p++){
						for(int q = y; q < y+8; q++){
							if(i == x && j == y)
								continue;
							else if (i == x+2 && j == y+1)
								continue;
							else if (i == x+2 && j == y)
								continue;
							else if (i == x+1 && j == y+2)
								continue;
							else if (i == x+1 && j == y+1)
								continue;
							else if (i == x+1 && j == y)
								continue;
							else if (i == x && j == y+3)
								continue;
							else if (i == x && j == y+2)
								continue;
							else if (i == x && j == y +1)
								continue;
							else
								fin_img.at<Vec3b>(i,j)[2] = 0;
						}
					}
				}
			}
		}
	}
	
	return fin_img;
}

Mat rgbToHSI(Mat orig_image){
	Mat fin_img = orig_image.clone();
	for(int x = 0; x < fin_img.rows; x++){
		for(int y = 0; y < fin_img.cols; y++){
			fin_img.at<Vec3b>(x, y)[0] = 0;
			fin_img.at<Vec3b>(x, y)[1] = 0;
			fin_img.at<Vec3b>(x, y)[2] = 0;
		}
	}

	float r, g, b, h, s, i;
	
	for(int x = 0; x < orig_image.rows; x++){
		for(int y = 0; y < orig_image.cols; y++){
			//Vec3b &pix = orig_image.at<Vec3b>(x, y);
			
			b = orig_image.at<Vec3b>(x,y)[0];
			g = orig_image.at<Vec3b>(x,y)[1];
			r = orig_image.at<Vec3b>(x,y)[2];
			
			/*b = pix[0];
			g = pix[1];
			r = pix[2];
			*/

			float bgr = b + g + r;
			b /= bgr;
			g /= bgr;
			r /= bgr;
			
			h = 0.5 * ((r-g) + (r-b)) / sqrt(((r-g) * (r-g)) + ((r-b) * (g-b)));
			h = 180 * acos(h) / PI;
			if(b > g) h = 360 - h;
			h = 255 * h / 360;

			float min = std::min(r,std:: min(b, g));
			s = (1 - (3 * min / (b+g+r))) ;

			if(s < 0.00001)
				s = 0;
			else if(s > 0.99999)
				s = 1;
			
			i = bgr/3;
			
			//pix[0] = h;
			//pix[1] = s;
			//pix[2] = i;

			fin_img.at<Vec3b>(x,y)[0] = h;
			fin_img.at<Vec3b>(x,y)[1] = s;
			fin_img.at<Vec3b>(x,y)[2] = i;
		}
	}

	return fin_img;

}

Mat get_channel_intensity(Mat orig_image, int channel){
	Mat fin_img = orig_image.clone();
	for(int x = 0; x < fin_img.rows; x++){
		for(int y = 0; y < fin_img.cols; y++){
			fin_img.at<Vec3b>(x, y)[0] = 0;
			fin_img.at<Vec3b>(x, y)[1] = 0;
			fin_img.at<Vec3b>(x, y)[2] = 0;
		}
	}

	float r, g, b, h, s, i;
	
	for(int x = 0; x < orig_image.rows; x++){
		for(int y = 0; y < orig_image.cols; y++){
			
			b = orig_image.at<Vec3b>(x,y)[0];
			g = orig_image.at<Vec3b>(x,y)[1];
			r = orig_image.at<Vec3b>(x,y)[2];
			
			float bgr = b + g + r;
			i = bgr/3;
			
			fin_img.at<Vec3b>(x,y)[channel] = i;
		}
	}

	return fin_img;

}

int calculateXGradient(Mat img, int x, int y){
	double XGrad[3][3] = {
		{-1, -2, -1},
		{0, 0, 0},
		{1, 2, 1}
	};
	
	int gradient = XGrad[0][0] * img.at<Vec3b>(x-1,y-1)[2] + XGrad[0][1] * img.at<Vec3b>(x-1, y)[2] + XGrad[0][2] * img.at<Vec3b>(x-1, y+1)[2]
		     + XGrad[1][0] * img.at<Vec3b>(x, y-1)[2]  + XGrad[1][1] * img.at<Vec3b>(x, y)[2]   + XGrad[1][2] * img.at<Vec3b>(x, y+1)[2]
		     + XGrad[2][0] * img.at<Vec3b>(x+1,y-1)[2] + XGrad[2][1] * img.at<Vec3b>(x+1, y)[2] + XGrad[2][2] * img.at<Vec3b>(x+1, y+1)[2];

	return gradient;
}

int calculateYGradient(Mat img, int x, int y){
	double YGrad[3][3] = {
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1}
	};
	
	int gradient = YGrad[0][0] * img.at<Vec3b>(x-1,y-1)[2] + YGrad[0][1] * img.at<Vec3b>(x-1, y)[2] + YGrad[0][2] * img.at<Vec3b>(x-1, y+1)[2]
		     + YGrad[1][0] * img.at<Vec3b>(x, y-1)[2]  + YGrad[1][1] * img.at<Vec3b>(x, y)[2]   + YGrad[1][2] * img.at<Vec3b>(x, y+1)[2]
		     + YGrad[2][0] * img.at<Vec3b>(x+1,y-1)[2] + YGrad[2][1] * img.at<Vec3b>(x+1, y)[2] + YGrad[2][2] * img.at<Vec3b>(x+1, y+1)[2];

	return gradient;
}

Mat sobel(Mat img){

	Mat fin_img = img.clone();
	for(int x = 0; x < fin_img.rows; x++){
		for(int y = 0; y < fin_img.cols; y++){
			fin_img.at<Vec3b>(x, y)[0] = 0;
			fin_img.at<Vec3b>(x, y)[1] = 0;
			fin_img.at<Vec3b>(x, y)[2] = 0;
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

			fin_img.at<Vec3b>(x, y)[2] = total_grad;
		}
	}

	return fin_img;
}

int main(){

	string basel3_image = "../img/basel3.bmp";
	string building_image = "../img/Building1.bmp";
	string disk_image = "../img/Disk.bmp";
	
	//------------- Original Basel3_image -----------
	Mat basel3_img = imread(basel3_image, CV_LOAD_IMAGE_COLOR);
	namedWindow("f1", WINDOW_AUTOSIZE);
	imshow("f1", basel3_img);

	//------------- HSI Base13_image ----------------
	Mat hsi_img = rgbToHSI(basel3_img);
	namedWindow("HSI Image", WINDOW_AUTOSIZE);
	imshow("HSI Image", hsi_img);

	//------------- Red Basel3_image ----------------
	Mat red_img = get_channel_intensity(basel3_img, 2);
	namedWindow("Red Intensity Image", WINDOW_AUTOSIZE);
	imshow("Red Intensity Image", red_img);
	
	//------------- Blue Basel3_image ----------------
	Mat blue_img = get_channel_intensity(basel3_img, 0);
	namedWindow("Blue Intensity Image", WINDOW_AUTOSIZE);
	imshow("Blue Intensity Image", blue_img);
	
	//------------- Green Basel3_image ----------------
	Mat green_img = get_channel_intensity(basel3_img, 1);
	namedWindow("Green Intensity Image", WINDOW_AUTOSIZE);
	imshow("Green Intensity Image", green_img);
	
	//------------- DCT Base13_image ----------------
	Mat dct_img = DCTImage(red_img);
	namedWindow("DCT Image", WINDOW_AUTOSIZE);
	imshow("DCT Image", dct_img);
	
	//------------- DCT_DC Base13_image ----------------
	Mat dct_dc_img = DCTImage_DC(dct_img);
	namedWindow("DCT_DC Image", WINDOW_AUTOSIZE);
	imshow("DCT_DC Image", dct_dc_img);

	//------------- DCT_9 Base13_image ----------------
	Mat dct_9_img = DCTImage_9(dct_img);
	namedWindow("DCT_9 Image", WINDOW_AUTOSIZE);
	imshow("DCT_9 Image", dct_9_img);
	
	//------------- IDCT_DC Base13_image ----------------
	Mat idct_dc_img = IDCTImage(dct_dc_img);
	namedWindow("IDCT_DC Image", WINDOW_AUTOSIZE);
	imshow("IDCT_DC Image", idct_dc_img);

	//------------- IDCT_9 Base13_image ----------------
	Mat idct_9_img = IDCTImage(basel3_img);
	namedWindow("IDCT_9 Image", WINDOW_AUTOSIZE);
	imshow("IDCT_9 Image", idct_9_img);
		
	//------------- ROI Building_image ----------------
	Mat building_img = imread(building_image, CV_LOAD_IMAGE_COLOR);
	Mat building_hsi_img = rgbToHSI(building_img);
	Mat building_roi_img = sobel(building_hsi_img);
	namedWindow("Building ROI Image", WINDOW_AUTOSIZE);
	imshow("Building ROI Image", building_roi_img);
	
	//------------- ROI Building_image ----------------
	Mat disk_img = imread(disk_image, CV_LOAD_IMAGE_COLOR);
	Mat disk_hsi_img = rgbToHSI(disk_img);
	Mat disk_roi_img = sobel(disk_hsi_img);
	namedWindow("Disk ROI Image", WINDOW_AUTOSIZE);
	imshow("Disk ROI Image", disk_roi_img);
		
	waitKey();
	destroyAllWindows();

	
}	
