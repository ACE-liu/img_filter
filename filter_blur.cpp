// =====================================================================================
// 
//       Filename:  blur_detection.cpp
// 
//        Version:  1.0
//        Created:  04/24/2018 04:51:46 PM
//       Revision:  none
//       Compiler:  g++
// 
//         Author:  HAN LIN, lin.han@iim.ltd
//        Company:  IIM
// 
//    Description:  blur detection
// 
// =====================================================================================

#include "filter_blur.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

FilterBlur::FilterBlur()
    : dct_weights{	/* diagonal weighting */
                    8,7,6,5,4,3,2,1,
                    1,8,7,6,5,4,3,2,
                    2,1,8,7,6,5,4,3,
                    3,2,1,8,7,6,5,4,
                    4,3,2,1,8,7,6,5,
                    5,4,3,2,1,8,7,6,
                    6,5,4,3,2,1,8,7,
                    7,6,5,4,3,2,1,8
                    }
{
    dct_minvalue = 8;//smaller, easier to pass
    dct_max_histogram_value = 0.005f;
    dct_total_weight = 344.0f;
    dct_threshold = 0.45;
}

FilterBlur::FilterBlur(int dct_minvalue_, float dct_threshold_)
    : dct_weights{	/* diagonal weighting */
                    8,7,6,5,4,3,2,1,
                    1,8,7,6,5,4,3,2,
                    2,1,8,7,6,5,4,3,
                    3,2,1,8,7,6,5,4,
                    4,3,2,1,8,7,6,5,
                    5,4,3,2,1,8,7,6,
                    6,5,4,3,2,1,8,7,
                    7,6,5,4,3,2,1,8
                    }
{

    dct_minvalue = dct_minvalue_;
    dct_max_histogram_value = 0.005f;
    dct_total_weight = 344.0f;
    dct_threshold = dct_threshold_;
}
bool FilterBlur::filtBlurlImg(cv::Mat const& cv_img){

    cv::Mat img_bw;
    cv::cvtColor(cv_img, img_bw, CV_BGR2GRAY);
    float dct_blur= discreteCosineTransform(img_bw);
    if(dct_blur>dct_threshold){
        return true;
    }
    else{
        return false;
    }
}

void FilterBlur::dctUpdateHistogram(cv::Mat const& dblock, std::vector<int>& histogram){
    uchar *ptrImg = dblock.data;
    for(int y = 0; y < 8; y++){
        for(int x = 0; x <8; x++){
            if(std::abs(ptrImg[x]) > dct_minvalue){
                histogram[y*8+x]++;
            }
        }
        ptrImg+=8;
    }
}

float FilterBlur::discreteCosineTransform(cv::Mat const& cv_img){
    cv::Mat fimage, dblock;
    std::vector<int> histogram(64);
    cv_img.convertTo(fimage, CV_32FC1); // also scale to [0..1] range (not mandatory)
    for (int i = 0; i < (cv_img.rows/8)*8; i += 8){
        for (int j = 0; j < (cv_img.cols/8)*8; j+= 8){
            cv::Mat fblock =fimage(cv::Rect(j,i,8,8));
            cv::dct( fblock, dblock );
        std::cout<<dblock<<"\n"<<std::endl;
            dblock.convertTo(dblock,CV_8UC1);
            dctUpdateHistogram(dblock, histogram);
        }
    }

   std::cout<<"[";
   for(auto &p : histogram)
   {
       std::cout<<p<<",";
   }

   std::cout<<"]"<<std::endl;
    float blur = 0.0;
    std::cout <<"thro : "<<dct_max_histogram_value*histogram[0]<<std::endl;
    int count =0;
    for(int k = 0; k < 64; k++){
        if(histogram[k] < dct_max_histogram_value*histogram[0]){
            blur += dct_weights[k];
            ++count;
        }
    }
    std::cout <<" >thre point count: "<<count<<std::endl;
    std::cout <<"Blur value: "<<blur<<std::endl;
    blur /= dct_total_weight;
    std::cout <<"blur v::: "<<blur<<std::endl;
    return blur;
}

