#include <iostream>
#include <opencv2/opencv.hpp>
#include "GLCM.h"
#include "Classifier.h"

using namespace std;
using namespace cv;

int main()
{
    Classifier classifier("../binary_classification");
    classifier.GetTrainingData();
    classifier.InitProb();
    classifier.InitParam();

//    cout << classifier.CrossValidation(3) << endl;
//    classifier.Train();
//    cout << classifier.Predict("../wood_grass/train/1/120.jpg") << endl;
//    cout << classifier.Predict("../binary_classification/train/-1/E120.jpg") << endl;
//    cout << classifier.Predict("../binary_classification/test/9.bmp") << endl;
//    classifier.ProcessImg("../binary_classification/test/8.bmp", "../binary_classification/test/8_rst.jpg");
    classifier.ProcessImgByCover("../binary_classification/test/10.bmp", "../binary_classification/test/10_rst.jpg");
    return 0;
}