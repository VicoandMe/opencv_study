#include "imgDetect.h"
#include <string>
#include <sstream>
#include <fstream>
using namespace cv;

int ConvertCharArrayToInt(unsigned char* array, int LengthOfArray)
{
    if (LengthOfArray < 0)
    {
        return -1;
    }
    int result = static_cast<signed int>(array[0]);
    for (int i = 1; i < LengthOfArray; i++)
    {
        result = (result << 8) + array[i];
    }
    return result;
}

bool IsImageDataFile(unsigned char* MagicNumber, int LengthOfArray)
{
    int MagicNumberOfImage = ConvertCharArrayToInt(MagicNumber, LengthOfArray);
    if (MagicNumberOfImage == MAGICNUMBEROFIMAGE)
    {
        return true;
    }

    return false;
}

bool IsLabelDataFile(unsigned char *MagicNumber, int LengthOfArray)
{
    int MagicNumberOfLabel = ConvertCharArrayToInt(MagicNumber, LengthOfArray);
    if (MagicNumberOfLabel == MAGICNUMBEROFLABEL)
    {
        return true;
    }

    return false;
}

cv::Mat ReadData(std::fstream& DataFile, int NumberOfData, int DataSizeInBytes)
{
    cv::Mat DataMat;

    if (DataFile.is_open())
    {
        int AllDataSizeInBytes = DataSizeInBytes * NumberOfData;
        unsigned char* TmpData = new unsigned char[AllDataSizeInBytes];
        DataFile.read((char *)TmpData, AllDataSizeInBytes);
        DataMat = cv::Mat(NumberOfData, DataSizeInBytes, CV_8UC1,
                          TmpData).clone();
        delete [] TmpData;
        DataFile.close();
    }

    return DataMat;
}

cv::Mat ReadImageData(std::fstream& ImageDataFile, int NumberOfImages)
{
    int ImageSizeInBytes = 28 * 28;

    return ReadData(ImageDataFile, NumberOfImages, ImageSizeInBytes);
}

cv::Mat ReadLabelData(std::fstream& LabelDataFile, int NumberOfLabel)
{
    int LabelSizeInBytes = 1;

    return ReadData(LabelDataFile, NumberOfLabel, LabelSizeInBytes);
}

cv::Mat ReadImages(std::string& FileName)
{
    std::fstream File(FileName.c_str(), std::ios_base::in | std::ios_base::binary);

    if (!File.is_open())
    {
        return cv::Mat();
    }

    MNISTImageFileHeader FileHeader;
    File.read((char *)(&FileHeader), sizeof(FileHeader));

    if (!IsImageDataFile(FileHeader.MagicNumber, 4))
    {
        return cv::Mat();
    }

    int NumberOfImage = ConvertCharArrayToInt(FileHeader.NumberOfImages, 4);
    cout<<NumberOfImage<<endl;
    return ReadImageData(File, NumberOfImage);
}

cv::Mat ReadLabels(std::string& FileName)
{
    std::fstream File(FileName.c_str(), std::ios_base::in | std::ios_base::binary);

    if (!File.is_open())
    {
        return cv::Mat();
    }

    MNISTLabelFileHeader FileHeader;
    File.read((char *)(&FileHeader), sizeof(FileHeader));

    if (!IsLabelDataFile(FileHeader.MagicNumber, 4))
    {
        return cv::Mat();
    }

    int NumberOfImage = ConvertCharArrayToInt(FileHeader.NumberOfLabels, 4);

    return ReadLabelData(File, NumberOfImage);
}

int get_trainingImage() {
  string imgName = "train-images.idx3-ubyte";
  string labeName = "train-labels.idx1-ubyte";
  cv::Mat a = ReadImages(imgName);
  cv::Mat b = ReadLabels(labeName);
  cv::Mat M(28, 28, CV_8UC1,Scalar::all(0));

  fstream file1;
  file1.open("./info.txt",ios::out|ios::in);
  for(int num = 0; num < 60000; num++) {
    uchar* data = a.ptr<uchar>(num);
    for (int j = 0; j < 28; j++) {
      uchar* Mdata = M.ptr<uchar>(j);
      for (int i = 0; i < 28; i++) {
        Mdata[i] = data[j*28 + i];
      }
    }
    stringstream ss;
    ss << num;
    string n;
    ss>>n;
    cout<<n<<endl;
    imwrite("./trainImg/"+n+".jpg",M);
    file1 << "trainImg/" + n + ".jpg" + " " +"1 0 0 27 27\n";
    }
}
