//
// Created by Chen on 2018/1/28.
//

#ifndef GLCM_SVM_CLASSIFIER_H
#define GLCM_SVM_CLASSIFIER_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <dirent.h>
#include <fstream>
#include "svm.h"
#include "GLCM.h"

using namespace std;
using namespace cv;

class Classifier
{
public:
    Classifier(String _folderName = "../img/binary_classification/");
    ~Classifier();

    String folderName;
    vector<svm_node*> trainX;
    vector<double> trainY;

    vector<String> GetFolderList(String folderName);
    void GetTrainingData();
    void Train();
    int Predict(String path);
};


#endif //GLCM_SVM_CLASSIFIER_H
