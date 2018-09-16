#include <iostream>
#include <queue>
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
	cout << "Threshold Value: " << threshold << endl;
	
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

int connected_comp(Mat& img, int x, int y, uchar set_color, uchar change_color){

    queue<int> queue_x;
    queue<int> queue_y;
    queue_x.push(x);
    queue_y.push(y);

    int pixelCount = 0;

    while(!queue_x.empty() && !queue_y.empty()) {
        int i = queue_x.front();
        int j = queue_y.front();
        queue_x.pop();
        queue_y.pop();

        // Check if visited before continuing
        if(img.at<uchar>(i,j) != set_color){

		//increase size of found region
		pixelCount++;

		//set as visited
		img.at<uchar>(i,j) = set_color;

		//find surrounding components that have not been visited
		if(j+1 < img.cols && img.at<uchar>(i,j+1) == change_color){
			queue_x.push(i);
			queue_y.push(j+1);
		}
		if(j-1 >= 0 && img.at<uchar>(i,j-1) == change_color){
			queue_x.push(i);	
			queue_y.push(j-1);
		}
		if(i+1 < img.rows && img.at<uchar>(i+1,j) == change_color){
			queue_x.push(i+1);
			queue_y.push(j);
		}
		if(i-1 >= 0 && img.at<uchar>(i-1,j) == change_color){
			queue_x.push(i-1);
			queue_y.push(j);
		}
	}
    }
   
    //return size 
    return pixelCount;
}

Mat regionDetection(Mat& img){
	int min_size = INT_MAX;
	int max_size = 0;
	int min_x = -1;
	int min_y = -1;
	int max_x = -1;
	int max_y = -1;

	//color all regions with medium color (120)
	//find the smallest and largest regions
	for(int x = 0; x < img.rows; x++){
		for(int y = 0; y < img.cols; y++){
			if(img.at<uchar>(x,y) == 255){
				int total_pixels = connected_comp(img, x, y, 120, 255);
				
				if(total_pixels < min_size){
					min_size = total_pixels;
					min_x = x;
					min_y = y;
				}else if(total_pixels > max_size){
					max_size = total_pixels;
					max_x = x;
					max_y = y;
				}

			}	
		}
	}

	//color smallest and largest regions	
	connected_comp(img, min_x, min_y, 60, 120);
	connected_comp(img, max_x, max_y, 200, 120);
	
	return img;
}

int main(){

	string night_image = "../img/House_width_times4.bmp";
	string nyc_image = "../img/NYC_width_4times.bmp";
	string shape_image = "../img/shapes2.1.bmp";
	string panic_image = "../img/guide_8bits.bmp";

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
/*
	//-------------- Negative NYC Image ---------------
	neg_img = createNegImage(img);
	namedWindow("Negative NYC Image", WINDOW_AUTOSIZE);
	imshow("Negative NYC Image", neg_img);
	showHistogram(neg_img, "Negative NYC Histogram");
*/	
	//-------------- Enhanced NYC Image ---------------
	enhanced_img = enhanceImage(img);	
	namedWindow("Enhanced NYC Image", WINDOW_AUTOSIZE);
	imshow("Enhanced NYC Image", enhanced_img);
	showHistogram(enhanced_img, "Enhanced NYC Histogram");

	//------------- Original Shape Image --------------
	img = imread(shape_image, IMREAD_GRAYSCALE);
	namedWindow("Original Shape Image", WINDOW_AUTOSIZE);
	imshow("Original Shape Image", img);

	//------------ Binary Shape Image -----------------
	Mat bin_img = createBinaryImage(img);
	namedWindow("Binary Shape Image", WINDOW_AUTOSIZE);
	imshow("Binary Shape Image", bin_img);

	//------------ Shape Detection --------------------
	Mat region = bin_img.clone();
	region = regionDetection(region);
	namedWindow("Shape Detection Image", WINDOW_AUTOSIZE);
	imshow("Shape Detection Image", region);

	//------------ Original Panic Image ---------------
	img = imread(panic_image, IMREAD_GRAYSCALE);
	namedWindow("Original Panic Image", WINDOW_AUTOSIZE);
	imshow("Original Panic Image", img);

	//------------ Binary Panic Image -----------------
	bin_img = createBinaryImage(img);
	namedWindow("Binary Panic Image", WINDOW_AUTOSIZE);
	imshow("Binary Panic Image", bin_img);

	//------------ Panic Region Detection -------------
	region = bin_img.clone();
	region = regionDetection(region);
	namedWindow("Panic Detection Image", WINDOW_AUTOSIZE);
	imshow("Panic Detection Image", region);
	
	waitKey(0);
	cvDestroyAllWindows();

	return 0;
}
