#include <iostream>
#include <stdio.h>
#include "imgDetect.h"
#include "findNegativeSamples.h"
#include "Neural_network.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"
//#include "Neural_network.h"

using namespace std;
using namespace cv;

void detectAndDraw(Mat& img,
                   CascadeClassifier& cascade,
                   double scale);

string cascadeName = "./num_test.xml";

int cheack_classigier() {
  //get_trainingImage();
  //get_NegativeSamples();
  //get_NegativeSamples();
  Mat image;
  CascadeClassifier cascade, nestedCascade;//创建级联分类器对象
  double scale = 1.3;
 //    image = imread("obama_gray.bmp",1);
  image = imread("num_test.jpg",1);
  cvtColor(image,image,CV_BGR2GRAY);
  for(int i = 0; i < image.cols;i++) {
    uchar *p =image.ptr<uchar>(i);
    for(int t = 0; t < image.rows; t++) {
     p[t] = 255 - p[t];
    }
  }
  if( !cascade.load( cascadeName ) )//从指定的文件目录中加载级联分类器
  {
          cerr << "ERROR: Could not load classifier cascade" << endl;
          return 0;
     }

     if( !image.empty() )//读取图片数据不能为空
     {
         detectAndDraw( image, cascade, scale );
         waitKey(0);
     }

     return 0;
}

void detectAndDraw(Mat& img,
                   CascadeClassifier& cascade,
                   double scale) {
    int i = 0;
    double t = 0;
    vector<Rect> nums;
    const static Scalar colors[] =  { CV_RGB(0,0,255),
           CV_RGB(0,128,255),
           CV_RGB(0,255,255),
           CV_RGB(0,255,0),
           CV_RGB(255,128,0),
           CV_RGB(255,255,0),
           CV_RGB(255,0,0),
           CV_RGB(255,0,255)} ;//用不同的颜色表示不同的人
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );
    //cvtColor( img, gray, CV_BGR2GRAY );
    resize( img, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    //equalizeHist( smallImg, smallImg );
    imwrite("temp.jpg",smallImg);
    cascade.detectMultiScale(smallImg, nums,1.1,2,
                            0|CV_HAAR_SCALE_IMAGE,
                            Size(5,5));
    cout<<nums.size()<<endl;
    for( vector<Rect>::const_iterator r = nums.begin(); r != nums.end(); r++, i++ )
    {
         //Mat smallImgROI;
         vector<Rect> nestedObjects;
         Point pt_1;
         Point pt_2;
         Scalar color = colors[i%8];
         //int radius;
         //center.x = cvRound((r->x + r->width*0.5)*scale);//还原成原来的大小
         //center.y = cvRound((r->y + r->height*0.5)*scale);
         //radius = cvRound((r->width + r->height)*0.25*scale);
         //circle( img, center, radius, color, 3, 8, 0 );
         pt_1.x = r->x;
         pt_1.y = r->y;
         pt_2.x = r->x + r->width;
         pt_2.y = r->y + r->height;
         rectangle(img, pt_1, pt_2, color);
         //smallImgROI = smallImg(*r);
     }
     cv::imwrite( "result.jpg", img );
     cout<<"end"<<endl;
}

int main() {
  printf("1: get negative sample\n2: get training image\n3: get training image lable\n4: check classifier\n5: Training Neural_network\n6: check Neural_network\n7: get testImage_info\n8: Test Neural_network\n");
  printf("enter: ");
  int e;
  while(1) {
  cin>>e;
  if(e == 1) {
    get_NegativeSamples();
  } else if (e == 2) {
    get_trainingImage();
  } else if (e == 3) {
    get_trainingImageLable();
  } else if (e == 4) {
    cheack_classigier();
  } else if (e == 5) {
    getTrainImageName();
    inputtra();
  } else if (e == 6) {
    check_Neural_network();
  } else if (e == 7) {
    get_testImage();
    get_testImageLable();
  } else if (e == 8) {
    getTestImageName();
    inputtest();
    check_Neural_network();
  }
  }
}
