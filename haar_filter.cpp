#include <iostream>
#include "haar_filter.h"

using namespace std;
using namespace cv;


haar_filter::haar_filter()
{
     threshold=35;
     MinZero=0.05;
}

void haar_filter::getHaarWavelet(const Mat &src, Mat &dst)
{
    int height = src.size().height;
    int width = src.size().width;
    dst.create(height, width, CV_32F);

    Mat horizontal = Mat::zeros(height, width, CV_32F);
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width/2; j++)
        {
            float meanPixel = (src.at<float>(i,2*j) + src.at<float>(i,2*j+1)) / 2;
            horizontal.at<float>(i, j) = meanPixel;
            horizontal.at<float>(i, j + width/2) = src.at<float>(i, 2*j) - meanPixel;
        }
    }
    for(int i = 0; i < height/2; i++)
    {
        for(int j = 0; j < width; j++)
        {
            float meanPixel = (horizontal.at<float>(2*i,j) + horizontal.at<float>(2*i+1,j)) / 2;
            dst.at<float>(i, j) = meanPixel;
            dst.at<float>(i + height/2, j) = horizontal.at<float>(2*i, j) - meanPixel;
        }
    }

    horizontal.release();

}

void haar_filter::getEmax(const Mat &src, Mat &dst, int scale)
{
    int height = src.size().height;
    int width = src.size().width;
    int h_scaled = height / scale;
    int w_scaled = width / scale;
    dst.create(h_scaled, w_scaled, CV_32F);

    for(int i = 0; i<h_scaled; i++)
    {
        for(int j = 0; j<w_scaled; j++)
        {
            double maxValue;
            minMaxLoc(src(Rect(scale*j, scale*i, scale, scale)), NULL, &maxValue);
            dst.at<float>(i,j) = float(maxValue);
        }
    }
}

bool haar_filter::filtBlurlImg(const Mat &cv_img)
{
    int height0 = cv_img.size().height;
    int width0 = cv_img.size().width;
    cv::Mat img_test;
    cv::cvtColor(cv_img, img_test, CV_BGR2GRAY);
    img_test.convertTo(img_test, CV_32FC1);

    std::cout<<"channels: "<<img_test.channels()<<std::endl;

    int height = ceilf(float(height0) / 16) * 16;
    int width = ceilf(float(width0) / 16) * 16;
    Mat img = Mat::zeros(height, width, CV_32F);
    Mat temp = img(Rect(0, 0, width0, height0));
    img_test.copyTo(img(Rect(0, 0, width0, height0)));
    Mat level1;
    getHaarWavelet(img, level1);
    Mat level2;
    getHaarWavelet(level1(Rect(0, 0, width/2, height/2)), level2);
    Mat level3;
    getHaarWavelet(level2(Rect(0, 0, width/4, height/4)), level3);

    // Step2
    Mat HL1, LH1, HH1, Emap1;
    pow(level1(Rect(width/2, 0, width/2, height/2)), 2.0, HL1);
    pow(level1(Rect(0, height/2, width/2, height/2)), 2.0, LH1);
    pow(level1(Rect(width/2, height/2, width/2, height/2)), 2.0, HH1);
    sqrt(HL1 + LH1 + HH1, Emap1);
    Mat HL2, LH2, HH2, Emap2;
    pow(level2(Rect(width/4, 0, width/4, height/4)), 2.0, HL2);
    pow(level2(Rect(0, height/4, width/4, height/4)), 2.0, LH2);
    pow(level2(Rect(width/4, height/4, width/4, height/4)), 2.0, HH2);
    sqrt(HL2 + LH2 + HH2, Emap2);
    Mat HL3, LH3, HH3, Emap3;
    pow(level3(Rect(width/8, 0, width/8, height/8)), 2.0, HL3);
    pow(level3(Rect(0, height/8, width/8, height/8)), 2.0, LH3);
    pow(level3(Rect(width/8, height/8, width/8, height/8)), 2.0, HH3);
    sqrt(HL3 + LH3 + HH3, Emap3);

    // Step3
    Mat Emax1, Emax2, Emax3;
    getEmax(Emap1, Emax1, 8);
    getEmax(Emap2, Emax2, 4);
    getEmax(Emap3, Emax3, 2);

    // Algorithm2: blur detection scheme
    // Step1 (Alegorithm 1)
    // Step2
    int m = Emax1.size().height;
    int n = Emax1.size().width;
    int Nedge = 0;
    Mat Eedge = Mat::zeros(m, n, CV_32F);
    for(int i = 0; i<m; i++)
    {
        for(int j = 0; j<n; j++)
        {
            if(Emax1.at<float>(i,j)>threshold || Emax2.at<float>(i,j)>threshold || Emax3.at<float>(i,j)>threshold)
            {
                ++Nedge;
                Eedge.at<float>(i,j) = 1.0;
            }
        }
    }
    // Step3 (Rule2)
    int Nda = 0;
    for(int i = 0; i<m; i++)
    {
        for(int j = 0; j<n; j++)
        {
            float tempEmax2 = Emax2.at<float>(i,j);
            if(Eedge.at<float>(i,j) > 0.1 && Emax1.at<float>(i,j)>tempEmax2 && tempEmax2 >Emax3.at<float>(i,j))
            {
                ++Nda;
            }
        }
    }
    // Step4 (Rule3,4)
    int Nrg = 0;
    Mat Eedge_Gstep_Roof = Mat::zeros(m, n, CV_32F);
    for(int i = 0; i<m; i++)
    {
        for(int j = 0; j<n; j++)
        {
            float tempEmax1 = Emax1.at<float>(i,j);
            float tempEmax2 = Emax2.at<float>(i,j);
            float tempEmax3 = Emax3.at<float>(i,j);
            if(Eedge.at<float>(i,j) > 0.1 && (tempEmax1<tempEmax2 && tempEmax2<tempEmax3 || tempEmax2>tempEmax1 && tempEmax2>tempEmax3))
            {
                ++Nrg;
                Eedge_Gstep_Roof.at<float>(i,j) = 1.0;
            }
        }
    }
    // Step5 (Rule5)
    int Nbrg = 0;
    for(int i = 0; i<m; i++)
    {
        for(int j = 0; j<n; j++)
        {
            if(Eedge_Gstep_Roof.at<float>(i,j) > 0.1 && Emax1.at<float>(i,j) < threshold)
            {
                ++Nbrg;
            }
        }
    }
    // Step6
    float Per = float(Nda) / Nedge;
    cout <<Per<<"........................"<<endl;
    if(Per > MinZero)
        return false;
    return true;
}



