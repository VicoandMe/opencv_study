#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <string>
#include <strstream>
#include <math.h>
#include <fstream>
using namespace std;
using namespace cv;

int get_NegativeSamples() {
  fstream file1;
  file1.open("./bg.txt",ios::out|ios::in);
  int countn = 0;
  for(int index = 1; index < 18; index++) {
  stringstream s1;
  s1<<index;
  string t;
  s1>>t;
  Mat imgsrc = imread(t + ".jpg");

  resize(imgsrc,imgsrc,Size(420,364),0,0,CV_INTER_LINEAR);
  cvtColor(imgsrc,imgsrc,CV_BGR2GRAY);

  for (int i = 0; i < 420-100;i = i + 5) {
    for (int j = 0; j < 364 - 100; j = j + 25) {
      Mat imgdst(imgsrc(Rect(i,j,100,100)));
      countn++;
      stringstream ss;
      ss<<countn;
      string a;
      ss>>a;
      cout<<countn<<endl;
      imwrite("./negative_samples/"+ a + ".jpg",imgdst);
      file1 << "negative_samples/" + a + ".jpg\n";
    }
  }
  countn++;
  }
  file1.close();
}
