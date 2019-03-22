#include "detector_face.h"
#include <opencv2/core/core.hpp>
#include "type_detection.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <boost/progress.hpp>
#include "system_config.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>  
 #include <opencv2/objdetect/objdetect.hpp>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include <iostream>
#include <caffe/common.hpp>
#include <sys/time.h>
#include <boost/filesystem.hpp>
#include "signal.h"

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>


using namespace std;
using namespace cv;

//#define  TASK_DIR    "/home/iim/Videos/"
#define  TASK_DIR    "/mnt/zhaoguang/"
#define  SKIP_GAP    0
//#define  OUTPUT_DIR  "/home/iim/Videos/"
#define  OUTPUT_DIR  "/mnt/liuliu_file/"
#define  TEST              1
#define  TEST_FILE         0
#define  OUTPUT_BIG_FILE   1

#define TEST_IMG           0

#define CUR_WIDTH          2048
#define CUR_HEIGHT         1536


#define  RECOGNIZE_WIDTH    80
#define  RECOGNIZE_HEIGHT   80

#define  BIG_SIZE           4000

#define out_put_face          1

enum detection_type{
	NO_FACE,
	FACE,
	GLASS_FACE
};

struct file_msg
{
    int count;
    int width;
    int height;
};


/*
 **dlib 相关参数
*/

static dlib::shape_predictor pose_model;

static const int total_landmark =68;

static bool ctrl_c_captured=false;
static void signal_callback(int sig)
{
    switch (sig) {
    case SIGINT:
        ctrl_c_captured = true;
        break;
    case SIGTERM:
        ctrl_c_captured = true;
        break;
    }
}

void init_dlib()
{
	std::string path ="/opt/ego/runtime/landmark/models/shape_predictor_68_face_landmarks.dat";
	dlib::deserialize(path)>>pose_model;
}

bool is_avi(const std::string &filename)
{
	if(filename.substr(filename.length()-4)==".avi")
	    return true;
	return false;
}

void load_file_from_dir(string rootPath,vector<string>&ret)
{
    namespace fs = boost::filesystem;
    fs::path fullpath (rootPath);
   
    if(!fs::exists(fullpath)){
        std::cout<<"the rootPath is not exit."<<endl;
        return;}

    fs::directory_iterator end_iter;  
    for(fs::directory_iterator iter(fullpath);iter!=end_iter;iter++){
        try{
            if (fs::is_directory( *iter ) ){
                load_file_from_dir(iter->path().string(),ret);
                std::cout<<"dir :"<<iter->path().string()<<endl;
            }else if(is_avi(iter->path().string())){
                ret.push_back(iter->path().string());
                std::cout << *iter << " is a file" << std::endl;
                std::cout<<"path :"<<iter->path().string()<<endl;
            }
        } catch ( const std::exception & ex ){
            std::cerr << ex.what() << std::endl;
            continue;
        }
    }
    return;
} 

bool check_if_has_glass(const perception::detection::DetectorFace* detector, cv::Mat &frame)
{
    assert(detector!=NULL);
    std::vector<cv::Rect>VecRect;
    boost::progress_timer t;
    detector->detectFace(frame, VecRect);
   // std::cout <<"test one frame cost: " << t.elapsed() << std::endl; 
	return VecRect.size()>0;
}
dlib::rectangle convert2dlibrect(cv::Rect const& face)
{
	return dlib::rectangle(face.x,face.y, face.width + face.x, face.height + face.y);
}

float caculateCurrentEntropy(Mat& hist, int threshold)
{      
	float BackgroundSum = 0, targetSum = 0;  
	const float*pDataHist = (float*)hist.ptr<float>(0);
	for (int i = 0; i < 256; i++){
		//累计背景值
		if (i < threshold){
			BackgroundSum += pDataHist[i];
		}
		else{//累计前景目标值
			targetSum += pDataHist[i];
		}
	}
	float BackgroundEntropy = 0, targetEntropy = 0;
	for (int i = 0; i < 256; i++){
		//计算背景熵
		if (i < threshold){
			if (pDataHist[i] == 0)
				continue;
			float ratiol = pDataHist[i] / BackgroundSum;
			//计算当前背景熵
			BackgroundEntropy += (-ratiol)*logf(ratiol);
		}
		else{//计算前景目标熵
			if (pDataHist[i] == 0)
				continue;
			float ratio2 = pDataHist[i] / targetSum;
			targetEntropy += (-ratio2)*logf(ratio2);
		}
	}
	return (targetEntropy + BackgroundEntropy);
}

float caculateCurrentEntropy(int* pDataHist, int threshold)
{
	float BackgroundSum = 0, targetSum = 0;
	//const float*pDataHist = (float*)hist.ptr<float>(0);
	for (int i = 0; i < 256; i++){
		//累计背景值
		if (i < threshold){
			BackgroundSum += pDataHist[i];
		}
		else{//累计前景目标值
			targetSum += pDataHist[i];
		}
	}
	float BackgroundEntropy = 0, targetEntropy = 0;
	for (int i = 0; i < 256; i++){
		//计算背景熵
		if (i < threshold){
			if (pDataHist[i] == 0)
				continue;
			float ratiol = pDataHist[i] / BackgroundSum;
			//计算当前背景熵
			BackgroundEntropy += (-ratiol)*logf(ratiol);
		}
		else{//计算前景目标熵
			if (pDataHist[i] == 0)
				continue;
			float ratio2 = pDataHist[i] / targetSum;
			targetEntropy += (-ratio2)*logf(ratio2);
		}
	}
	return (targetEntropy + BackgroundEntropy);
}

static int channels[1] = { 0 };
static int histSize[1] = { 256 };
static float pranges[2] = { 0, 256 };
static const float * ranges[1] = { pranges };
Mat maxEntropySegMentation(Mat& inputImg,bool black)
{
	//MatND hist;
	Mat result;
	float maxentropy = 0;
	int max_index = 0;
    cv::Mat inputImage =inputImg.clone();
	int hist[256];
	memset(hist,0,sizeof(hist));
	//计算直方图
	std::cout <<inputImage.cols<<" "<<inputImage.rows<<" "<<inputImage.channels()<<std::endl;
	for(int i=0;i<inputImage.rows;i++)
	{
		uchar *ptr =inputImage.ptr<uchar>(i);
		for(int j=0; j< inputImage.cols;j++)
		{
            int value =ptr[j];
			if(value<0||value>255)
			   std::cout<<"error value:"<<value<<std::endl;
			else 
			   hist[value]++;
		}
	}
	//calcHist(&inputImage, 1, channels,
	//	Mat(), hist, 1, histSize, ranges);

	//遍历分割阈值，并求取最大熵下的分割阈值
	for (int i = 0; i < 256; i++){
		float cur_entropy = caculateCurrentEntropy(hist, i);
		//取最大熵下的分割阈值
		if (cur_entropy > maxentropy) {
			maxentropy = cur_entropy;
			max_index = i;
		}
	}
	std::cout<<"max index: "<<max_index<<std::endl;
	if(black)
	{
	 cv::threshold(inputImage, inputImage, max_index, 255,
		CV_THRESH_BINARY_INV);
	}
	else
	  cv::threshold(inputImage,inputImage,max_index,255,CV_THRESH_BINARY);
	return inputImage;
}

float caculateCurrentEntropy1(int hist[][256],int data,int mean_data)
{
	assert(data<256&&mean_data<256);
	float BackgroundSum = 0, targetSum = 0;
	float BackgroundEntropy = 0, targetEntropy = 0;
	for(int i=0;i<data;i++)
	{
		for(int j=0; j<mean_data; j++)
		{
			BackgroundSum+=hist[i][j];
		}
	}
    for(int i=data;i<256;i++)
	{
		for(int j=mean_data; j<256; j++)
		{
			targetSum+=hist[i][j];
		}
	}
	//std::cout<< BackgroundSum <<" "<<targetSum<<std::endl;
	for(int i=0;i<data;i++)
	{
		for(int j=0; j<mean_data; j++)
		{
			if (hist[i][j] == 0)
				continue;
			float ratiol = hist[i][j] / BackgroundSum;
			//计算当前背景熵
			BackgroundEntropy += (-ratiol)*logf(ratiol);
		}
	}
	for(int i=data;i<256;i++)
	{
		for(int j=mean_data; j<256; j++)
		{
			if (hist[i][j] == 0)
				continue;
			float ratiol = hist[i][j] / targetSum;
			//计算当前背景熵
			targetEntropy += (-ratiol)*logf(ratiol);
		}
	}
    return BackgroundEntropy+targetEntropy;
}

int roi_width = 16;

Mat maxEntropySegMentation1(cv::Mat &img)
{
	Mat rtn;
	int hist[256][256]={0};
	std::cout <<"hist size :"<<sizeof(hist)<<std::endl;
    if(img.empty())
	    return rtn;
	int width =img.cols;
	int height =img.rows;	
	rtn =Mat::zeros(height,width,CV_8UC1);
	cv::Mat median_mat;
	//std::vector<std::pair<int ,int>>*status =new std::vector<std::pair<int ,int>>();
	//status->resize(width*height);
	cv::medianBlur(img,median_mat,3);
	cv::namedWindow("111",CV_WINDOW_NORMAL);
	cv::imshow("111",median_mat);
	cv::waitKey();
	for(int i=0; i<height; i++)
	{
		uchar *pdata = img.ptr<uchar>(i);
		uchar *mdata =median_mat.ptr<uchar>(i);
		for(int j=0;j<width; j++)
		{
			int data=pdata[j];
			int mean_data=mdata[j];
			hist[data][mean_data]++;
		}
	}    
	float maxentropy = 0;
	int max_index = 0;
	int max_mean_index=0;
	for(int i=0; i<256;i++)
	{	
		for (int j=0; j<256;j++)
		{
            float cur_entropy = caculateCurrentEntropy1(hist, i,j);
			std::cout<<cur_entropy<<std::endl;
		      //取最大熵下的分割阈值
		   if (cur_entropy > maxentropy) {
			  maxentropy = cur_entropy;
			  max_index = i;
			  max_mean_index=j;
		     }
		}
	}
	std::cout <<"max_index :"<<max_index <<" max_mean_Index: "<<max_mean_index<<std::endl;
	for(int i=0; i<height; i++)
	{
		uchar *odata = rtn.ptr<uchar>(i);
		uchar *pdata = img.ptr<uchar>(i);
		uchar *mdata =median_mat.ptr<uchar>(i);
		for(int j=0;j<width; j++)
		{
			if(pdata[j]<max_index&&mdata[j]<max_mean_index)
			      odata[j]=255;
			else
			      odata[j]=0;
		}
	}

	return rtn;
}
vector<Point> FindBigestContour(Mat src) {
	int imax = -1;
	int imaxcontour = -1;
	int total_size =src.cols*src.rows;
	std::vector<std::vector<Point> >contours;
	findContours(src, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	for (int i = 0; i<contours.size(); i++) {
		int itmp = contourArea(contours[i]);
		if (imaxcontour < itmp) {
			imax = i;
			imaxcontour = itmp;
		}
	}
	if(imax ==-1)
	{
		vector<Point> rtn;
		return rtn;
	}
	else 
	   return contours[imax];
}

static int ipa=0;
bool cas_has_glass(cv::Mat & img ,cv::Rect & face_rect)
{
	cv::Mat img_ =img.clone();
	cv::Mat grey;
	CascadeClassifier cascade;
	cascade.load("/opt/ego/opencv/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml");
	cv::cvtColor(img,grey,COLOR_BGR2GRAY);
	std::vector <cv::Rect> rect;

	cascade.detectMultiScale(grey, rect, 1.1, 3, 0);
	printf("检测到人脸个数：%d\n", rect.size()); 

	for (int i = 0; i < rect.size(); i++)
	{    //caffe   
		Point center;
		int radius;
		center.x = cvRound((rect[i].x + rect[i].width * 0.5));
		center.y = cvRound((rect[i].y + rect[i].height * 0.5));
		radius = cvRound((rect[i].width + rect[i].height) *0.25);
		circle(img_, center, radius, cv::Scalar(0,0,255), 2);
	}
    cv::imwrite("/home/iim/Videos/face_img.jpg",img_);
    if(rect.size()>0)
	   return true; 
} 

bool has_glass(cv::Mat& img,cv::Rect& face_rect)
{ 
	bool test_result=false;
	//cv::Mat img_;
	//img_ =img.clone();
	 std::vector < cv::Point2f > facial_landmark_points(total_landmark);
	dlib::cv_image < dlib::bgr_pixel > cimg(img);
    dlib::rectangle dlib_rect =convert2dlibrect(face_rect);
	dlib::full_object_detection shape = pose_model(cimg, dlib_rect);
	cv::Point2f point;
    for (int dots_idx = 0; dots_idx < total_landmark; dots_idx++)
    {
        dlib::point lmk = shape.part(dots_idx);
        point.x = lmk(0);
        point.y = lmk(1);
        facial_landmark_points[dots_idx] = point;
    }
//show
   //for(int i=0; i< facial_landmark_points.size(); i++)
  // {
    //   cv::circle(img_,facial_landmark_points[i],3,cv::Scalar(0,0,255),-1);
  // }
	//cv::namedWindow("face",CV_WINDOW_NORMAL);
	//cv::imshow("face",img_); 
	//cv::waitKey(); 
//show
    float min=facial_landmark_points[17].y;
	float max=facial_landmark_points[17].y;
	for(int i =18; i<=26 ;i++)
	{
         float value =facial_landmark_points[i].y;
		 if(value<min)
		     min=value;
		 else if(value > max)
		     max=value;		    
	}  	
    
	//float top =min - (max-min)>0?(min-(max-min)):min;
	float top =min;
    float bot =facial_landmark_points[33].y;
	

	float rect_top =face_rect.y;
	float rect_left=face_rect.x;
	float rect_bot =face_rect.y+face_rect.height;
	float rect_right =face_rect.x+face_rect.width;
	
	float cut_top=rect_top>top?rect_top:top;
	float cut_bot =rect_bot<bot?rect_bot:bot;
	float cut_left=rect_left;
	float cut_right =rect_right;



    
	cv::Rect2f cur_rect(cut_left,cut_top,cut_right-cut_left,cut_bot-cut_top);
    cv::Mat img_check=img(cur_rect).clone();
	cv::Mat img_check1=img_check.clone();
	cv::Mat img_check2=img_check.clone();
	cv::Mat img_check3=img_check.clone();



	float roi_b =facial_landmark_points[33].y;
	float roi_x =facial_landmark_points[33].x -roi_width/2;
	float roi_y= cut_top;
	cv::Rect2f roi_rect(roi_x,roi_y,roi_width,roi_b-roi_y);
	cv::Mat img_roi_=img(roi_rect).clone();
	cv::Mat img_roi_gray,soble_img_gray,img_roi_gray_bin;
	//cv::imwrite("/home/iim/Videos/img_roi.jpg",img_roi_);
	GaussianBlur(img_roi_, img_roi_, Size(3, 3), 0, 0, BORDER_DEFAULT);
	cv::cvtColor(img_roi_,img_roi_gray,COLOR_BGR2GRAY);
	int g_nStructElementSize2 = 1; 
	cv::Mat element2 = getStructuringElement(MORPH_RECT,
		Size(2 * g_nStructElementSize2 + 1, 2 * g_nStructElementSize2 + 1),
		Point(g_nStructElementSize2, g_nStructElementSize2));
	erode(img_roi_gray, img_roi_gray, element2);
	// 膨胀
	dilate(img_roi_gray, img_roi_gray, element2);
	//cv::imwrite("/home/iim/Videos/img_roi_gray.jpg",img_roi_gray);
	cv::Sobel(img_roi_gray,soble_img_gray,CV_8U,0,1,3,1,0);
  // cv::imwrite("/home/iim/Videos/soble_img.jpg",soble_img_gray);
   float mean_value_ =cv::mean(soble_img_gray)[0];
   int count_=0;
   int totol_count=0;
   int height =soble_img_gray.rows*4/5;
    for(int i=0; i<height; i++)
	{
		count_=0;
		for(int j=0;j<soble_img_gray.cols; j++)
		{
			int min_=i+6>height?height:(i+6);
			for(int k=i;k<min_;k++)
			{
				uchar *pdata = soble_img_gray.ptr<uchar>(k);
				int data=pdata[j];
				if(data>50)  
				{
			       count_++;
				   break;
				}
			}
		} 
		if(count_>=soble_img_gray.cols-3)
		   totol_count++;
	} 
	//std::cout <<soble_img_gray<<std::endl;
	//std::cout <<"mean:"<<mean_value_<<" rows :"<<soble_img_gray.rows<<"cols :"<<soble_img_gray.cols<<std::endl;
//	std::cout <<"totol_count:"<<totol_count<<std::endl;

	if(totol_count>0)
	   return true;
	else
	   return false;
  // cv::threshold(img_roi_gray,img_roi_gray_bin,)



	cv::Mat img_gray(img_check.rows,img_check.cols,CV_8UC1);
	cv::Mat img_test(img_check.rows,img_check.cols,CV_8UC1);
	 GaussianBlur(img_check, img_check, Size(3, 3), 0, 0, BORDER_DEFAULT);
	cv::cvtColor(img_check,img_gray,COLOR_BGR2GRAY);
	//cv::imwrite("/home/iim/Videos/gray_img.jpg",img_gray);
	float mean_value =cv::mean(img_gray)[0];

	//std::cout<<img_gray<<std::endl;
    cv::Point  interest1 =cv::Point(facial_landmark_points[39].x-cut_left,facial_landmark_points[39].y-cut_top);
	 cv::Point  interest2 =cv::Point(facial_landmark_points[42].x-cut_left,facial_landmark_points[42].y-cut_top);
	  cv::Point  interest3 =cv::Point(facial_landmark_points[18].x-cut_left,facial_landmark_points[18].y-cut_top);
	   cv::Point  interest4 =cv::Point(facial_landmark_points[25].x-cut_left,facial_landmark_points[25].y-cut_top);
	     cv::Point  interest5 =cv::Point(facial_landmark_points[19].x-cut_left,facial_landmark_points[19].y-cut_top);
		   cv::Point  interest6 =cv::Point(facial_landmark_points[24].x-cut_left,facial_landmark_points[24].y-cut_top);


    //cv::Mat fil_data,fil_data1;

      //cv::equalizeHist(face_data,fil_data1);
	//cv::imwrite("/home/iim/Videos/fil_face1.jpg",fil_data1);

	 //cv::equalizeHist(img_gray,fil_data);
	//cv::imwrite("/home/iim/Videos/img_gray_equal.jpg",fil_data);
  
/*
	cv::Point  left_1 =cv::Point(facial_landmark_points[36].x-cut_left,facial_landmark_points[36].y-cut_top);
	cv::Point  left_2 =cv::Point(facial_landmark_points[37].x-cut_left,facial_landmark_points[37].y-cut_top);
	cv::Point  left_3 =cv::Point(facial_landmark_points[38].x-cut_left,facial_landmark_points[38].y-cut_top);
	cv::Point  left_4 =cv::Point(facial_landmark_points[39].x-cut_left,facial_landmark_points[39].y-cut_top);
	 cv::Point  left_5 =cv::Point(facial_landmark_points[40].x-cut_left,facial_landmark_points[40].y-cut_top);
	cv::Point  left_6 =cv::Point(facial_landmark_points[41].x-cut_left,facial_landmark_points[41].y-cut_top);
   std::vector<cv::Point> left_points;
   left_points.push_back(left_1);
   left_points.push_back(left_2);
   left_points.push_back(left_3);
   left_points.push_back(left_4);
   left_points.push_back(left_5);
   left_points.push_back(left_6);
	cv::RotatedRect box=cv::fitEllipse(left_points);

	cv::Point  right_1 =cv::Point(facial_landmark_points[42].x-cut_left,facial_landmark_points[42].y-cut_top);
	cv::Point  right_2 =cv::Point(facial_landmark_points[43].x-cut_left,facial_landmark_points[43].y-cut_top);
	cv::Point  right_3 =cv::Point(facial_landmark_points[44].x-cut_left,facial_landmark_points[44].y-cut_top);
	cv::Point  right_4 =cv::Point(facial_landmark_points[45].x-cut_left,facial_landmark_points[45].y-cut_top);
	 cv::Point  right_5 =cv::Point(facial_landmark_points[46].x-cut_left,facial_landmark_points[46].y-cut_top);
	cv::Point  right_6 =cv::Point(facial_landmark_points[47].x-cut_left,facial_landmark_points[47].y-cut_top);
   std::vector<cv::Point> right_points;
   right_points.push_back(right_1);
   right_points.push_back(right_2);
   right_points.push_back(right_3);
   right_points.push_back(right_4);
   right_points.push_back(right_5);
   right_points.push_back(right_6);
	cv::RotatedRect right_box=cv::fitEllipse(right_points);
*/

	float intx_min =interest3.x;
	float intx_max =interest4.x;
	float intx_mid1 =interest1.x;
	float intx_mid2 =interest2.x;
	float int_mid=(intx_mid1+intx_mid2)/2;

    float inty_min =interest5.y>interest6.y?interest6.y:interest6.y;
	float inty_max =interest1.y>interest2.y?interest1.y:interest2.y;
//check w

	int g_nStructElementSize = 3; 
	cv::Mat element = getStructuringElement(MORPH_RECT,
		Size(2 * g_nStructElementSize + 1, 2 * g_nStructElementSize + 1),
		Point(g_nStructElementSize, g_nStructElementSize));

   img_gray.copyTo(img_test);

  // ellipse(img_test	,box,cv::Scalar(0),-1,8);
   //ellipse(img_test	,right_box,cv::Scalar(0),-1,8);

/*
   cv::Mat soble_img,bin_soble;
   cv::Sobel(img_test,soble_img,CV_8U,0,1,3,1,0);
   cv::imwrite("/home/iim/Videos/soble_img.jpg",soble_img);
   mean_value =cv::mean(soble_img)[0];
   std::cout <<"sobel mean2 W:"<<mean_value<<std::endl;
     cv::threshold(soble_img,bin_soble,25,255,THRESH_BINARY);
     cv::imwrite("/home/iim/Videos/bin_soble_img.jpg",bin_soble);
	 soble_img=maxEntropySegMentation(soble_img,false);
	 cv::imwrite("/home/iim/Videos/max_soble_img.jpg",soble_img);
*/
     cv::equalizeHist(img_test,img_test);
    mean_value =cv::mean(img_test)[0];
   cv::imwrite("/home/iim/Videos/img_gray_w.jpg",img_test);
   cv::threshold(img_test,img_test,mean_value,255,THRESH_TOZERO);
   img_test=maxEntropySegMentation(img_test,false);

    
   vector<Point> bigestcontrour = FindBigestContour(img_test); 

   
   int left_mid_count=0;
    int right_mid_count=0;
   int min_count =0;
   int max_count=0;
   int glass_count=0;
   int total_size=0;
   int total_count=0;
   float percent=0;
	vector<vector<Point> > controus;
	controus.push_back(bigestcontrour);   
#if TEST_IMG
    string img_name1,img_name2;
	img_name1.append("/home/iim/Videos/gray_white_max");
	img_name1.append(to_string(ipa));
	img_name1.append(".jpg");
	img_name2.append("/home/iim/Videos/gray_white");
	img_name2.append(to_string(ipa));
	img_name2.append(".jpg");
	cv::drawContours(img_check, controus, 0, Scalar(0, 0, 255), 3);
    cv::imwrite(img_name1,img_check);
	cv::imwrite(img_name2,img_test);
#endif
    //double len =arcLength(bigestcontrour,true);
	//double per=len/contourArea(bigestcontrour) ;
	float var =contourArea(bigestcontrour)/(img_test.cols*img_test.rows);
	//std::cout <<"len:"<<len<<"per:"<<per<<" "<< contourArea(bigestcontrour)<<" "<<img_gray.cols*img_gray.rows<<" "<<var<<std::endl;

	//img_test.release();
    total_size=bigestcontrour.size();
	if(total_size>0)
	{
	for(int i=0;i<bigestcontrour.size();i++)
	{
		cv::Point p= bigestcontrour[i];
		float px =p.x;
		float py=p.y;
		if(px>intx_mid1&&px<int_mid)
		{
		    left_mid_count++;
			if(py>inty_min&&py<inty_max)
		        glass_count++;
		}
		else if(px>int_mid&&px<intx_mid2)
		{
		    right_mid_count++;
			if(py>inty_min&&py<inty_max)
		        glass_count++;
		}
		else if(px>intx_min) 
		    min_count++;
		else if(px<intx_max)
		    max_count++;
	}
	 total_count =left_mid_count+right_mid_count+min_count+max_count+glass_count;
	 percent =total_count/(float)total_size;
	if(left_mid_count>3&&right_mid_count>3&&glass_count>3&&(min_count>3||max_count>3))
    {
		if(var>0.1&&var<0.5) 
	       test_result=true;
		else if(percent>0.50)
		    test_result=true;
	}	
	}
	if(test_result)
	{
		ipa++;
	   return true;
	}
	std::cout<<"DATA: "<< min_count<<" "<<left_mid_count<<" "<<right_mid_count<<" "<<max_count<<" "<<glass_count<<" "<<total_size<<" "<<percent<<std::endl;     
   // return test_result;


//black
    //ellipse(img_gray	,box,cv::Scalar(255),-1,8);
   //ellipse(img_gray	,right_box,cv::Scalar(255),-1,8);
   cv::equalizeHist(img_gray,img_gray);
    mean_value =cv::mean(img_gray)[0];
	std::cout <<"mean3 B:"<<mean_value<<std::endl;
   cv::imwrite("/home/iim/Videos/img_gray_b.jpg",img_gray);
   for(int i=0; i<img_gray.rows; i++)
	{
		uchar *pdata = img_gray.ptr<uchar>(i);
		for(int j=0;j<img_gray.cols; j++)
		{
			int data=pdata[j];
			if(data>mean_value)  
			   pdata[j]=255;
		} 
	}  
	img_gray=maxEntropySegMentation(img_gray,true);
    int g_nStructElementSize1 = 1; 
	cv::Mat element1 = getStructuringElement(MORPH_RECT,
		Size(2 * g_nStructElementSize1 + 1, 2 * g_nStructElementSize1 + 1),
		Point(g_nStructElementSize1, g_nStructElementSize1));
	// 膨胀
	dilate(img_gray, img_gray, element1);
	// 腐蚀
	erode(img_gray, img_gray, element1);
	bigestcontrour = FindBigestContour(img_gray);
     
	controus.clear();
	controus.push_back(bigestcontrour);
#if TEST_IMG
	img_name1.assign("/home/iim/Videos/gray_black_max");
	img_name1.append(to_string(ipa));
	img_name1.append(".jpg");
	img_name2.assign("/home/iim/Videos/gray_black");
	img_name2.append(to_string(ipa));
	img_name2.append(".jpg");
	cv::drawContours(img_check1, controus, 0, Scalar(0, 0, 255), 3);
    cv::imwrite(img_name1,img_check1);
	cv::imwrite(img_name2,img_gray);
#endif
	var =contourArea(bigestcontrour)/(img_gray.cols*img_gray.rows);
	// len =arcLength(bigestcontrour,true);
	// per=len/contourArea(bigestcontrour) ;
//	std::cout <<"len:"<<len<<"per:"<<per<<" "<< contourArea(bigestcontrour)<<" "<<img_gray.cols*img_gray.rows<<" "<<var<<std::endl;
	//img_gray.release();
	 left_mid_count=0;
	 right_mid_count=0;
    min_count =0;
    max_count=0;
    glass_count=0;
	total_size=bigestcontrour.size();
	if(total_size>0){
	for(int i=0;i<bigestcontrour.size();i++)
	{
		cv::Point p= bigestcontrour[i];
		float px =p.x;
		float py=p.y;
		if(px>intx_mid1&&px<int_mid)
		{
		    left_mid_count++;
			if(py>inty_min&&py<inty_max)
		        glass_count++;
		}
		else if(px>int_mid&&px<intx_mid2)
		{
		    right_mid_count++;
			if(py>inty_min&&py<inty_max)
		        glass_count++;
		}
		else if(px>intx_min) 
		    min_count++;
		else if(px<intx_max)
		    max_count++;
	}
	total_count =left_mid_count+right_mid_count+min_count+max_count+glass_count;
	percent =total_count/(float)total_size;
	if(left_mid_count>3&&right_mid_count>3&&glass_count>3&&(min_count>3||max_count>3))
    {
		if(var>0.1&&var<0.5) 
	       test_result=true;
		else if(percent>0.50)
		    test_result=true;
	}	
	}
	ipa++;
std::cout<<"DATA: "<< min_count<<" "<<left_mid_count<<" "<<right_mid_count<<" "<<max_count<<" "<<glass_count<<" "<<total_size<<" "<<percent<<std::endl;  
   // std::cout<<"DATA: "<< min_count<<" "<<mid_count<<" "<<max_count<<" "<<glass_count<<" "<<total_size<<" "<<percent<<std::endl;   
    return test_result;
} 
static int glass_img_index=0;
static int img_index=0;

void save_img(cv::Mat& img,std::string path1,int& index)
{
	std::string path;
	path+=OUTPUT_DIR;
	path +=path1;
	if(!boost::filesystem::exists(path))
  {
    boost::filesystem::create_directories(path);
  }
  string filename =path+"glass"+to_string(index)+".jpg";
  std::cout<<filename<<std::endl;
  cv::imwrite(filename,img);
  index++;
}
detection_type check_if_has_recognize_glass(const perception::detection::DetectorFace* detector, cv::Mat &frame)
{
    assert(detector!=NULL); 
    std::vector<cv::Rect>VecRect;
    boost::progress_timer t;
    detector->detectFace(frame, VecRect);
   // std::cout <<"test one frame cost: " << t.elapsed() << std::endl; 
   if(VecRect.size()==0)
       return NO_FACE;
	std::cout << "face cout :"<<VecRect.size()<<std::endl;
   for(int i =0;i<VecRect.size();i++)
   {    
	   if(VecRect[i].width>RECOGNIZE_WIDTH&&VecRect[i].height>RECOGNIZE_HEIGHT)
	   {
	   //cv::Mat cur_face =(frame(VecRect[i])).clone();
	    if(has_glass(frame,VecRect[i]))
	    {
	       return GLASS_FACE;
	    }
	   }
   }
   return FACE;
}

bool check_if_save_cur_frame(const perception::detection::DetectorFace *detector, cv::Mat &frame)
{
	assert(detector!=NULL);

    return false;

}


string get_next_filename(string file, int index)
{
	string filename = file.substr(file.find_last_of('/')+1);
	int len = filename.find_last_of('.');
	string new_filename =OUTPUT_DIR+ filename.substr(0, len).append("_piece_") + to_string(index)+".avi";
#if TEST
	std::cout << "file Name is: " << filename << endl;
	std::cout << "new file Name is: " << new_filename << endl;
#endif
    return new_filename;
}

string get_next_output_filename(int index,bool if_glass=false)
{
	string fileName;
	if(if_glass)
	    fileName=fileName+OUTPUT_DIR+"glass_output"+to_string(index)+".avi";
	else
	    fileName=fileName+OUTPUT_DIR+"output"+to_string(index)+".avi";
#if TEST
    std::cout<<"file Name is: "<<fileName<<endl;
#endif
    return fileName;
}


void checkfileLists(const perception::detection::DetectorFace* mDetector_, const std::vector<std::string>& video_files
                    , const int check_width, const int check_height, std::vector<file_msg>& log_msg)
{
    cv::Mat frame;
    int cur_frame_count = 0;
    bool status = false;
    int last_start_loc = -1;
    int last_end_loc = -1;
    string next_fileName;
    string glass_next_fileName;
    int glass_index = 0;
    int fourcc;
    double fps;
    int index = 0;
    int fileIndex = 1;
    VideoWriter writer;
    VideoWriter glass_writer;
    int glass_cur_frame_count = 0;
    bool glass_status = false;
    bool quit = false;
    int width;
    int height;
    int next_width, next_height;
    std::vector<std::string>other_files;
    int dealed = 0;
    int total = video_files.size();
    status = false;
    if (total == 0)
        return;
    for (int i = 0; i < video_files.size(); i++) {
#if !OUTPUT_BIG_FILE
        status = false;
        cur_frame_count = 0;
        index = 0;
#endif
        if (ctrl_c_captured)
            break;
        std::cout << "total file: " << video_files.size() << "\n" << "has dealed with: " << i << "\n" << "cur_frame_count: " <<
                  cur_frame_count << endl;
        string file = video_files[i];
        VideoCapture capture(file);
        if (!capture.isOpened()) {
            std::cout << file << "fail to open!" << endl;
            continue;
        }
        fileIndex++;
        fourcc = (int)capture.get(CAP_PROP_FOURCC);
        fps = capture.get(CAP_PROP_FPS);
        width = capture.get(CAP_PROP_FRAME_WIDTH);
        height = capture.get(CAP_PROP_FRAME_HEIGHT);
        if (check_width != width || check_height != height) {
            cout <<check_width<<" "<<check_height<<std::endl;
            cout << file << ":" << "width :" << width << "height :" << height << endl;
            capture.release();
            next_width = width;
            next_height = height;
            other_files.push_back(file);
            continue;
        } else
            dealed++;
        while (capture.read(frame)) {
            /* if(cur_frame_count%(SKIP_GAP+1)!=0)
            {
            cur_frame_count++;
            continue;
            }*/
            if(ctrl_c_captured)
                break;
#if OUTPUT_BIG_FILE
            detection_type result = check_if_has_recognize_glass(mDetector_, frame);
            if (result == FACE || result == GLASS_FACE) {
#if out_put_face
                //  save_img(frame,"img/",img_index);
                if (status == false) {
                    next_fileName = get_next_output_filename(index++);
                    cur_frame_count = 0;
                    if (!writer.open(next_fileName, fourcc, fps, frame.size())) {
                        std::cout << next_fileName << " can not be opened!" << endl;
                        continue;
                    }
                    status = true;
                    writer << frame;
                    cur_frame_count++;
                    //last_start_loc=cur_frame_count;
                } else {
                    if (writer.isOpened()) {
                        writer << frame;
                        cur_frame_count++;
                        std::cout << "cur_frame_count :" << cur_frame_count << endl;
                    } else
                        std::cout << "why video writer do not opened!" << endl;
                    if (cur_frame_count == BIG_SIZE) {
                        writer.release();
                        status = false;
                        std::cout << next_fileName << " BIG FILE has successfully saved." << endl;
                    }
                }
#endif
            }
            if (result == GLASS_FACE) {
                //save_img(frame,"img_glass/",glass_img_index);
                if (glass_status == false) {
                    glass_next_fileName = get_next_output_filename(glass_index++, true);
                    glass_cur_frame_count = 0;
                    if (!glass_writer.open(glass_next_fileName, fourcc, fps, frame.size())) {
                        std::cout << glass_next_fileName << " can not be opened!" << endl;
                        continue;
                    }
                    glass_status = true;
                    glass_writer << frame;
                    glass_cur_frame_count++;
                    //last_start_loc=cur_frame_count;
                } else {
                    if (glass_writer.isOpened()) {
                        glass_writer << frame;
                        glass_cur_frame_count++;
                        std::cout << "glass_cur_frame_count :" << cur_frame_count << endl;
                    } else
                        std::cout << "glass why video writer do not opened!" << endl;
                    if (glass_cur_frame_count == BIG_SIZE) {
                        glass_writer.release();
                        glass_status = false;
                        std::cout << glass_next_fileName << "glass BIG FILE has successfully saved." << endl;
                    }
                }
            }
            //std::cout<<"file :"<<file<<"\n"<<"fourcc:"<<fourcc<<endl;
#else
            if (check_if_has_glass(mDetector_, frame)) {
                if (status == false) {
                    next_fileName = get_next_filename(file, index++);
                    if (!writer.open(next_fileName, fourcc, fps, frame.size())) {
                        std::cout << next_fileName << " can not be opened!" << endl;
                        continue;
                    }
                    status = true;
                    writer << frame;
                    //last_start_loc=cur_frame_count;
                } else {
                    if (writer.isOpened())
                        writer << frame;
                    else
                        std::cout << "why video writer do not opened!" << endl;
                }
            } else {
                if (status == true) {
                    status = false;
                    // last_end_loc=cur_frame_count;
                    if (writer.isOpened())
                        writer.release();
                }
            }
            cur_frame_count++;
#endif
        }
#if !OUTPUT_BIG_FILE
        if (writer.isOpened())
            writer.release();
#endif
        capture.release();
        cout << "di " << to_string(i) << "file has dealed witch" << endl;
    }
    if (writer.isOpened())
        writer.release();
    if (glass_writer.isOpened())
        glass_writer.release();
    file_msg cur_msg;
    cur_msg.width = check_width;
    cur_msg.height = check_height;
    cur_msg.count = dealed;
    log_msg.push_back(cur_msg);
    if (dealed < total)
        checkfileLists(mDetector_, other_files, next_width, next_height, log_msg);
}

#if TEST_IMG

int main(int argc, char** argv)
{
    cv::Mat frame;
    std::string file_name = "/home/iim/Videos/test1.jpg";
    frame = imread(file_name);
    if (frame.empty()) {
        std::cout << "frame is empty!";
        return 0;
    }
    caffe::Caffe::SetDevice(0);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    init_dlib();
    std::string path = iim_ego::core::SystemConfig::getConstPtrInstance()->getRuntimePath() + "/detection";
    perception::detection::DetectorFace* mDetector_ = new perception::detection::DetectorFace(path + "/settings/face.xml",
            path);
    detection_type result = check_if_has_recognize_glass(mDetector_, frame);
    if (result == FACE) {
        std::cout << "has detect face." << std::endl;
    } else if (result == GLASS_FACE) {
        std::cout << "has detect glass." << std::endl;
    } else
        std::cout << "detect nothing." << std::endl;
    return 0;
}

#else
int main(int argc, char** argv)
{
    signal(SIGINT, signal_callback);
    signal(SIGTERM, signal_callback);
    vector<string> video_files;
#if TEST
    std::cout << "start test." << endl;
#endif
    load_file_from_dir(TASK_DIR, video_files);
#if TEST_FILE
    std::cout << "test now:" << video_files.size() << endl;
    return 0;
#endif
    std::vector<file_msg> log_msg;
    caffe::Caffe::SetDevice(0);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    std::string path = iim_ego::core::SystemConfig::getConstPtrInstance()->getRuntimePath() + "/detection";
    perception::detection::DetectorFace* mDetector_ = new perception::detection::DetectorFace(path + "/settings/face.xml",
            path);
    init_dlib();
    checkfileLists(mDetector_, video_files, CUR_WIDTH, CUR_HEIGHT, log_msg);
    std::cout << "处理完成！" << std::endl;
    std::cout << "总视频数： " << video_files.size() << "  其中:" << std::endl;
    for (int i = 0; i < log_msg.size(); i++) {
        file_msg msg = log_msg[i];
        printf("帧率为%dx%d的视频数量为： %d", msg.width, msg.height, msg.count);
    }
    delete mDetector_;
    return 0;
}
#endif
