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
    svm_problem prob;
    svm_parameter param;
    vector<svm_node*> trainX;
    vector<double> trainY;

    vector<String> GetFolderList(String folderName);
    void GetTrainingData();
    void InitProb();
    void InitParam();
    double CrossValidation(int foldNum);
    void Train();
    int Predict(String path);
    int Predict(svm_model* model, const Mat img);
    void ProcessImg(String srcPath, String rstPath);
    void GetIntegralImage(InputArray _src, OutputArray _intImg, int power = 1);
    void ProcessImgByCover(String srcPath, String rstPath);
};


#endif //GLCM_SVM_CLASSIFIER_H
