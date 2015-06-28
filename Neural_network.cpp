#include "Neural_network.h"

vector<tra> tradata;
vector<tra> testdata;

vector<string> TrainImgName;
vector<string> TestImgName;

node hNN[10][10];
node1 NN1[10];
node2 NN[10];

using namespace cv;

void update1 (int i, double e, double y) {
  for (int a = 0; a < 10; a++) {
     NN[i].w[a] = NN[i].w[a] + 0.1*e*y*(1-y)*NN1[i].w[a];
  }
}

void update2 (int i, double e, tra temp) {
  for (int a = 0; a < 10; a++) {
    for (int c = 0; c < 8; c++) {
      for (int n = 0; n < 8; n++) {
        hNN[i][a].w[n][c] =hNN[i][a].w[n][c] + 0.1*e*NN[i].w[a]*NN1[i].w[a]*(1-NN1[i].w[a])*temp.a[n][c];
      }
    }
  }
}

void single_train(int i, tra temp) {
  double d;
  double e;
  double y;

  if (i == temp.value) {
    d = 1;
  } else {
    d = 0;
  }

  for (int a = 0; a < 10; a++) {
    for (int c = 0; c < 8; c++) {
      for (int n = 0; n < 8; n++) {
        NN1[i].w[a] = NN1[i].w[a] + temp.a[n][c]*hNN[i][a].w[n][c];
      }
    }
    NN1[i].w[a] = 1/(1+exp(0 - NN1[i].w[a]));
  }
  for(int a = 0; a < 10; a++) {
    y = y + NN1[i].w[a] * NN[i].w[a];
  }
  y = 1/(1+exp(0-y));
  e = d - y;
  update1(i,e,y);
  update2(i,e,temp);
}

void train(tra temp) {
  for (int i = 0; i < 10; i++) {
    single_train(i,temp);
  }
}

void getTrainImageName() {
  ifstream in("./info.txt");
  if (!in.is_open()) {
    cout << "Error opening file";
  }

  string tdata;
  string get = "";
  while(getline(in,tdata)) {
    int index = 0;
    get = "";
    for(;index < tdata.length(); index++) {
      if(tdata[index] != ' ') {
        get.push_back(tdata[index]);
      } else {
        break;
      }
    }
    TrainImgName.push_back(get);
  }
}

vector<int> trainlables;

bool inputtra() {
  ifstream in("./trainingLabels.txt");
  string lables;
  getline(in,lables);
  int l = 0;
  for (int i = 0; i < lables.length(); i++) {
    if (lables[i] == ',') {
      continue;
    } else {
      l = lables[i] -'0';
    }
    trainlables.push_back(l);
  }

  for (int index = 0; index < TrainImgName.size(); index++) {
    Mat TrainImg = imread(TrainImgName[index]);
    //cout<<TrainImg.cols<<endl;
    resize(TrainImg, TrainImg, Size(8,8), 0, 0, CV_INTER_LINEAR);
    tra temp;
    for (int i = 0; i < 8; i++) {
      uchar *p = TrainImg.ptr<uchar>(i);
      for (int t = 0; t < 8; t++) {
        if((int)p[t] > 200) {
          temp.a[i][t] = 1;
          } else {
          temp.a[i][t] = 0;
          }
      }
    }
    temp.value = trainlables[index];
    tradata.push_back(temp);
  }
  cout <<tradata.size() <<endl;
  for(int t = 0; t < 10; t++) {
  for(int i = 0; i < tradata.size(); i++) {
    cout <<"round: "<<t<<" "<< "training image:" << i << endl;
    train(tradata[i]);
  }
  }
}

void getTestImageName() {
  ifstream in("./testInfo.txt");
  if (!in.is_open()) {
    cout << "Error opening file";
  }

  string tdata;
  string get = "";
  while(getline(in,tdata)) {
    int index = 0;
    get = "";
    for(;index < tdata.length(); index++) {
      if(tdata[index] != ' ') {
        get.push_back(tdata[index]);
      } else {
        break;
      }
    }
    TestImgName.push_back(get);
  }
}

vector<int> Testlables;

bool inputtest() {
  ifstream in("./testLabels.txt");
  string lables;
  getline(in,lables);
  int l = 0;
  for (int i = 0; i < lables.length(); i++) {
    if (lables[i] == ',') {
      continue;
    } else {
      l = lables[i] -'0';
    }
    Testlables.push_back(l);
  }

  for (int index = 0; index < TestImgName.size(); index++) {
    Mat TrainImg = imread(TestImgName[index]);
    //cout<<TrainImg.cols<<endl;
    resize(TrainImg, TrainImg, Size(8,8), 0, 0, CV_INTER_LINEAR);
    tra temp;
    for (int i = 0; i < 8; i++) {
      uchar *p = TrainImg.ptr<uchar>(i);
      for (int t = 0; t < 8; t++) {
        if((int)p[t] > 200) {
          temp.a[i][t] = 1;
          } else {
          temp.a[i][t] = 0;
          }
      }
    }
    temp.value = Testlables[index];
    testdata.push_back(temp);
  }
  cout<<testdata.size()<<endl;
}

double get(int i, tra temp) {
	double d;
    double e;
    double y;

	if(i == temp.value) {
		d = 1;
	} else {
		d = 0;
	}

	for(int a = 0; a < 10;a++) {
	  for(int c = 0; c < 8; c++) {
		for(int n = 0; n < 8; n++) {
			NN1[i].w[a] = NN1[i].w[a] + temp.a[n][c]*hNN[i][a].w[n][c];
		}
	  }
	  NN1[i].w[a] = 1/(1+exp(0-NN1[i].w[a]));
    }

	for(int a = 0; a < 10; a++) {
		y = y + NN1[i].w[a]*NN[i].w[a];
	}
	y = 1/(1+exp(0-y));
  return y;
}

bool check(tra temp) {
	int n = 0;
	double v[10];
	for(int i = 0; i < 10; i++) {
		v[i] = get(i,temp);
	}
	double a = v[0];
	for(int i = 1; i < 10; i++) {
		if(v[i] > a) {
			a = v[i];
			n = i;
		}
	}
	if(n == temp.value) {
		return true;
	} else {
		return false;
	}
}
void check_Neural_network() {
  double r = 0;
  double f = 0;
  	for(int i = 0; i < testdata.size(); i++) {
		if(check(testdata[i])) {
			r++;
		} else {
			f++;
		}
	}
	cout<<"Size_of_trainning_set:"<<tradata.size()<<endl;
	cout<<"Size_of_testing_set:"<<testdata.size()<<endl;
	cout<<"Accuracy:"<<(r/(r+f))*100<<"%"<<endl;
}
