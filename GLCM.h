//
// Created by Chen on 2018/1/27.
//
#ifndef GLCM_SVM_GLCM_H
#define GLCM_SVM_GLCM_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

struct GLCMFeature_t
{
    double entropy;
    double homogeneity;
    double contrast;
    double ASM;
    double correlation;
    int featureNum = 5;
};

class GLCM
{
public:
    GLCM(InputArray _srcImg, int _GLCMClass = -1);
    GLCM(String path, int _GLCMClass = -1);
    GLCM();
    ~GLCM();

    Mat GLCMMat;
    Mat srcImg;
    int GLCMClass;
    double maxPixVal;
    GLCMFeature_t GLCMFeature;
    void Init(InputArray _srcImg, int _GLCMClass = -1);
    void Init(String path, int _GLCMClass = -1);
    void CalGLCM(int angle = 0, int offset = 1, bool norm = true);
    void CalFeature();
    vector<double> GetFeaturesByAngle(int *angleList, int angleNum);
};


#endif //GLCM_SVM_GLCM_H
