#include <iostream>
#include <opencv2/opencv.hpp>
#include "GLCM.h"
#include "Classifier.h"

using namespace std;
using namespace cv;

int main()
{
    Classifier classifier("../img/binary_classification/");
    classifier.GetTrainingData();
    classifier.Train();
    cout << classifier.Predict("../img/binary_classification/-1/E550.jpg") << endl;

    return 0;
}