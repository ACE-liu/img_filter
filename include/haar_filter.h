#ifndef HAAR_FILTER_H
#define HAAR_FILTER_H

#include <opencv2/opencv.hpp>


class haar_filter
{
public:
    haar_filter();
    bool filtBlurlImg(cv::Mat const& cv_img);

private:
    float MinZero;
    float threshold;
    void getHaarWavelet(const cv::Mat &src, cv::Mat &dst);
    void getEmax(const cv::Mat &src, cv::Mat &dst, int scale);

};
#endif // HAAR_FILTER_H
