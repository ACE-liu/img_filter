#include <iostream>


#include "include/filter_blur.h"
#include "include/haar_filter.h"
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <unistd.h>
#include <sstream>
#include <algorithm>
#include <sys/stat.h>
#include <unistd.h>


#define serDir  "/home/liuliu/Desktop/data"
#define saveDir "/home/liuliu/Desktop/checkdata/"


#define TEST_ONE_FILE 1

using namespace cv;
using namespace std;

static FilterBlur blurcheck;
static haar_filter haarcheck;


bool check_if_blur(const cv::Mat &img,bool if_haar)
{
    if(if_haar)
        return haarcheck.filtBlurlImg(img);
    return blurcheck.filtBlurlImg(img);
}



bool is_img(const std::string &filename)
{
    std::string subStr =filename.substr(filename.length()-4);
    if(subStr==".jpg"||subStr==".png")
	    return true;
	return false;
}

void load_file_from_dir(std::string rootPath,std::vector<std::string>&ret)
{
    namespace fs = boost::filesystem;
    fs::path fullpath (rootPath);
    if(!fs::exists(fullpath)){
        std::cout<<"the rootPath is not exit."<<std::endl;
        return;}

    fs::directory_iterator end_iter;  
    for(fs::directory_iterator iter(fullpath);iter!=end_iter;iter++){
        try{
            if (fs::is_directory( *iter ) ){
                load_file_from_dir(iter->path().string(),ret);
                std::cout<<"dir :"<<iter->path().string()<<std::endl;
            }else if(is_img(iter->path().string())){
                ret.push_back(iter->path().string());
                std::cout << *iter << " is a file" << std::endl;
                std::cout<<"path :"<<iter->path().string()<<std::endl;
            }
        } catch ( const std::exception & ex ){
            std::cerr << ex.what() << std::endl;
            continue;
        }
    }
    return;
} 


bool mkdir(const std::string& path)
{
   if(access(path.c_str(),0)==0)
       return true;
   return mkdir(path.c_str(),0777) ==0;
}

void check_filedir_allfile()
{
    std::vector<std::string> check_file_list;
    load_file_from_dir(serDir, check_file_list);
    std::string savePath =saveDir;
    std::string saveBlurPath =savePath+"blur/";
    std::string saveotherPath =savePath+"other/";
    mkdir(savePath);
    mkdir(saveBlurPath);
    mkdir(saveotherPath);
//    std::cout<<"total file cout: "<<check_file_list.size()<<std::endl;
    sleep(3);
    int blurconut=0,othercount=0;
    int count=0;
    for(std::string &file :check_file_list)
    {
      if(++count%10)
      {
          std::cout<<"check file num: "<<count<<"............."<<std::endl;
      }
	  cv::Mat img =cv::imread(file);
     if(img.empty())
     {
	     continue;
     }
     if(check_if_blur(img,false))
     {
       std::string filepath =file;
       std::size_t pos= filepath.find_last_of("/\\");
       filepath.replace(pos,1,"_");
       pos= filepath.find_last_of("/\\");
       if(pos!=std::string::npos)
           filepath =filepath.substr(pos+1);
       blurconut++;
        cv::imwrite(saveBlurPath+filepath,img);
}
     else
     {
         std::string filepath =file;
         std::size_t pos= filepath.find_last_of("/\\");
         filepath.replace(pos,1,"_");
         pos= filepath.find_last_of("/\\");
         if(pos!=std::string::npos)
             filepath =filepath.substr(pos+1);
         othercount++;
         cv::imwrite(saveotherPath+filepath,img);
     }
}
    std::cout<<"test result:\n";
    std::cout<<"total file cout: "<<check_file_list.size()<<std::endl;
    std::cout<<"blur file count: "<<blurconut<<std::endl;
    std::cout<<"other file count: "<<othercount<<std::endl;
}

void check_laplas(const cv::Mat &imageSource)
{
    Mat imageGrey;

    cvtColor(imageSource, imageGrey, CV_RGB2GRAY);
    Mat imageSobel;

//     IplImage *img = &(IplImage(imageGrey));

    Laplacian(imageGrey, imageSobel, CV_8U);

//    Sobel(imageGrey, imageSobel, CV_16U, 1, 1);

    //图像的平均灰度
    double meanValue = 0.0;
    cv::Scalar meanVec,stdVec;
    meanStdDev(imageSobel,meanVec,stdVec);
    //double to string
//    stringstream meanValueStream;
//    string meanValueString;
//    meanValueStream << meanValue;
//    meanValueStream >> meanValueString;
//    meanValueString = "Articulation(Laplacian Method): " + meanValueString;
//    putText(imageSource, meanValueString, Point(20, 50), CV_FONT_HERSHEY_COMPLEX, 0.8, Scalar(255, 255, 25), 2);
//    imshow("Articulation", imageSource);
    std::cout <<"Laplacian  mean value: "<<meanVec[0]<<std::endl;
     std::cout <<"Laplacian  std value: "<<stdVec[0]<<std::endl;
    waitKey();

}

int main(int argc, char **argv) {
    std::cout << "start check...." << std::endl;
    std::string imgName;
#if TEST_ONE_FILE
    if(argc >1)
    {
      imgName =argv[1];
}
else
{
	std::cout <<"input img name..."<<std::endl;
	return -1;
}
  cv::Mat img =cv::imread(imgName);
  if(img.empty())
  {
    std::cout <<"quit..."<<std::endl;
    return -1;   
}

  if(check_if_blur(img,true))
  {
    std::cout<<"img is blur...."<<std::endl;
}
else 
	std::cout <<"img is ok..."<<std::endl;

  check_laplas(img);
#else
    check_filedir_allfile();
#endif
    
    return 0;
}
