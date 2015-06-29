
#ifndef TEST_H
#define TEST_H
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <math.h>
#include <time.h>
#include <stdlib.h>

#include <fstream>
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/opencv.hpp"
//#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#define g 6
#define rate
using namespace std;

struct tra {
  int a[28][28];
  int value;
};

static unsigned int seed = 0;

struct node {
  double w[28][28];
  node() {
    for(int i =  0; i < 28; i++) {
        for(int a = 0; a < 28; a++) {
          srand(seed++);
          w[i][a] = (rand()%20)/10-0.5;
        }
    }
  }
};


struct node1 {
	double w[10];
	node1() {
		for(int i =  0; i < 10; i++) {
			srand(seed++);
			w[i] = (rand()%10)/10-0.5;
		}
	}
};

struct node2 {
	double w[10];
	node2() {
		for(int i =  0; i < 10; i++) {
			srand(seed++);
			w[i] = (rand()%10)/10-0.5;
		}
	}
};

void update1(int i, double e, double y);
void update2(int i, double e, tra temp);
void single_train(int i, tra temp);
void train(tra temp);
bool inputtra();
bool inputtest();
void getTrainImageName();
void getTestImageName();
double get(int i, tra temp);
bool check(tra temp);
void check_Neural_network();
void train_Neural_network();
#endif
