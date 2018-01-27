//
// Created by Chen on 2018/1/27.
//
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#ifndef GLCM_SVM_GLCM_H
#define GLCM_SVM_GLCM_H

struct GLCMFeature_t
{
    double entropy;
    double homogeneity;
    double contrast;
    double ASM;
    double correlation;
};

class GLCM
{
public:
    GLCM(InputArray _src, int GLCM_class = 256);
    GLCM(String path, int GLCM_class = 256);
    ~GLCM();


//private:
    Mat GLCMMat;
    Mat srcImg;
    int GLCMClass;
    GLCMFeature_t GLCMFeature;
    void CalGLCM(int angle = 0, int offset = 1, bool norm = true);
    void CalFeature();
};


#endif //GLCM_SVM_GLCM_H
