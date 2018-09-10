#include <iostream>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

int histogram[256] = {0};

//generate histogram int array given image
void createHistogram(Mat img){

	for(int x = 0; x < 256; x++){
		histogram[x] = 0;
	}
	
	for(int x = 0; x < img.rows; x++){
		for(int y = 0; y < img.cols; y++){
			histogram[img.at<uchar>(x,y)] += 1;
		}
	}
}

//displays histogram
void showHistogram(Mat img, string win_name){
	createHistogram(img);
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound(double(hist_w)/256);

	Mat histogram_image(hist_h, hist_w, CV_8UC1, Scalar(255, 255, 255));
	
	//find max intensity
	int max = 0;
	for(int x = 0; x < 256; x++){
		if(max < histogram[x]){
			max = histogram[x];
		}
	}

	//normalize histogram
	for(int x = 0; x < 256; x++){
		histogram[x] = ((double)(histogram[x])/max) * histogram_image.rows;
	}
	
	//plot graph
	for(int x = 0; x < 256; x++){
		line(histogram_image, Point(bin_w*(x), hist_h), Point(bin_w*(x), hist_h - histogram[x]), Scalar(0,0,0), 1, 8, 0);
	}

	namedWindow(win_name, WINDOW_AUTOSIZE);
	imshow(win_name, histogram_image);
}

//creates negative image
Mat createNegImage(Mat img){
	Mat negImg = img.clone();
	
	for(int x = 0; x < img.rows; x++){
		for(int y = 0; y < img.cols; y++){
			negImg.at<uchar>(x,y) = 255 - img.at<uchar>(x,y);
		}
	}
	return negImg;
	
}

//create binary image
Mat createBinaryImage(Mat img){
	int total_pixels = img.rows * img.cols;
	createHistogram(img);
	int threshold = 0;
	
	//find threshold by calculating the averages of pixels
	for(int x = 0; x < 256; x++){
		threshold += x * histogram[x];
	}
	threshold = threshold/total_pixels;
	
	Mat bin_img = img.clone();
	for(int x = 0; x < bin_img.rows; x++){
		for(int y = 0; y < bin_img.cols; y++){
			if(bin_img.at<uchar>(x,y) < threshold)
				bin_img.at<uchar>(x,y) = 0;
			else
				bin_img.at<uchar>(x,y) = 255;
				
		}
	}
	
	return bin_img;
}

//create enhanced image
Mat enhanceImage(Mat img){
	int total_pixels = img.rows * img.cols;

	createHistogram(img);

	//calculate intensity probability of image
	float intensity_prob[256] = {0};
	for(int x = 0; x < 256; x++){
		intensity_prob[x] = (double)histogram[x]/total_pixels;
	}

	//calculate cumilative freq
	float cum_array[256] = {0};
	cum_array[0] = histogram[0];	
	for(int x = 1; x < 256; x++){
		cum_array[x] = (histogram[x]) + cum_array[x-1]; 
	}

	//calculate cumulative distribution probability
	float cdf[256] = {0};
	for(int x = 0; x < 256; x++){
		cdf[x] = cum_array[x]/total_pixels;
	}

	//multiply cdf by number of bins to calculate enhanced pixel
	float final_array[256] = {0};
	for(int x = 0; x < 256; x++){
		final_array[x] = floor(cdf[x]*255);
	}

	//generate enhanced image
	Mat enhanced_img = img.clone();
	for(int x = 0; x < img.rows; x++){
	        for(int y = 0; y < img.cols; y++){
       			enhanced_img.at<uchar>(x,y) = (final_array[img.at<uchar>(x,y)]);
		}
	}

	return enhanced_img;

}

int main(){

	string night_image = "../img/House_width_times4.bmp";
	string nyc_image = "../img/NYC_width_4times.bmp";
	string shape_image = "../img/shapes2.1.bmp";
	

	//-------------- Original Night Image ---------------
	Mat img = imread(night_image, IMREAD_GRAYSCALE);
	namedWindow("Original Night Image", WINDOW_AUTOSIZE);
	imshow("Original Night Image", img);
	showHistogram(img, "Original Night Histogram");

	//-------------- Negative Night Image ---------------
	Mat neg_img = createNegImage(img);
	namedWindow("Negative Night Image", WINDOW_AUTOSIZE);
	imshow("Negative Night Image", neg_img);
	showHistogram(neg_img, "Negative Night Histogram");
	
	//-------------- Enhanced Night Image ---------------
	Mat enhanced_img = enhanceImage(img);	
	namedWindow("Enhanced Night Image", WINDOW_AUTOSIZE);
	imshow("Enhanced Night Image", enhanced_img);
	showHistogram(enhanced_img, "Enhanced Night Histogram");
	
	//-------------- Original NYC Image ---------------
	img = imread(nyc_image, IMREAD_GRAYSCALE);
	namedWindow("Original NYC Image", WINDOW_AUTOSIZE);
	imshow("Original NYC Image", img);
	showHistogram(img, "Original NYC Histogram");

	//-------------- Negative NYC Image ---------------
	neg_img = createNegImage(img);
	namedWindow("Negative NYC Image", WINDOW_AUTOSIZE);
	imshow("Negative NYC Image", neg_img);
	showHistogram(neg_img, "Negative NYC Histogram");
	
	//-------------- Enhanced NYC Image ---------------
	enhanced_img = enhanceImage(img);	
	namedWindow("Enhanced NYC Image", WINDOW_AUTOSIZE);
	imshow("Enhanced NYC Image", enhanced_img);
	showHistogram(enhanced_img, "Enhanced NYC Histogram");

	//------------- Original Shape Image --------------
	img = imread(shape_image, IMREAD_GRAYSCALE);
	namedWindow("Original Shape Image", WINDOW_AUTOSIZE);
	imshow("Original Shape Image", img);
	showHistogram(img, "Original Shape Histogram");

	//------------ Binary Shape Image -----------------
	Mat bin_img = createBinaryImage(img);
	namedWindow("Binary Shape Image", WINDOW_AUTOSIZE);
	imshow("Binary Shape Image", bin_img);
	showHistogram(bin_img, "Binary Shape Histogram");
	

	waitKey(0);
	cvDestroyAllWindows();

	return 0;
}
