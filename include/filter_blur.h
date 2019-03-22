/**
 *          @file  filter_blur.h
 *
 *         @brief  filter for blur
 *
 *       @version  1.0
 *          @date  16/08/2018 12:21:50 PM
 *        @author  Lin HAN (Robotics), lin.han@iim.ltd
 * 
 * @section Description
 * 
 *       Revision:  none
 *       Compiler:  g++
 *        Company:  IIM
 * 
 * @section Description
 *
 * -->describe more here<--
 *
 */

#ifndef _FILTER_BLUR_H_
#define _FILTER_BLUR_H_

#include <opencv2/core/core.hpp>
#include <vector>


/**
 * @brief class to filt blur image in registration.
 *
 */
class FilterBlur
{
public:
    /**
     * @brief constructor with default parameter.
     *
     */
    FilterBlur();
    /**
     * @brief constructor with given parameter.
     *
     * @param dct_minvalue: The smaller the easier to pass, default is 8.
     * @param dct_threshold: default is 0.45.
     */
    FilterBlur(int dct_minvalue, float dct_threshold);
    /**
     * @brief destructor.
     *
     */
    ~FilterBlur() = default;
    /**
     * @brief If image is blur, return true, else return false.
     *
     * @param cv_img: Input image
     * @return bool: True = blur
     */
    bool filtBlurlImg(cv::Mat const& cv_img);
private: 
    /**
     * @brief conduct discrete cosine transform and return the blur extent.
     *
     * @param cv_img: input image
     * @return float: blur extent
     */
    float discreteCosineTransform(cv::Mat const& cv_img);
    /**
     * @brief update 8*8 block histogram.
     *
     * @param dblock: 8*8 block
     * @param histogram: 8*8 histogram
     */
    void dctUpdateHistogram(cv::Mat const& dblock, std::vector<int>& histogram);

    int dct_minvalue; /**< parameter used in DCT, update histogram */
    float dct_max_histogram_value; /**< parameter used in DCT, adjustable */
    const float dct_weights[64]; /**< parameter used in DCT, diagonal weighting */
    float dct_total_weight; /**< parameter used in DCT */
    float dct_threshold;/**< parameter used in DCT, compare with blur extent */
};


#endif //!_FILTER_BLUR_H_
