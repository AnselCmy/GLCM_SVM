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

    cout << classifier.CrossValidation(3) << endl;
//    classifier.Train();
//    cout << classifier.Predict("../binary_classification/train/-1/E120.jpg") << endl;
//    classifier.ProcessImg("../binary_classification/test/5.bmp", "../binary_classification/test/5_rst.jpg");
    return 0;
}