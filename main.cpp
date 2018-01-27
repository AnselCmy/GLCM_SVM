#include <iostream>
#include <opencv2/opencv.hpp>
#include "GLCM.h"

using namespace std;
using namespace cv;

int main()
{
    GLCM glcm("../img/5.bmp", 16);
    glcm.CalGLCM(90);
    glcm.CalFeature();
    cout << glcm.GLCMFeature.entropy << endl;
    cout << glcm.GLCMFeature.homogeneity << endl;
    cout << glcm.GLCMFeature.contrast << endl;
    cout << glcm.GLCMFeature.ASM << endl;
    cout << glcm.GLCMFeature.correlation << endl;
    return 0;
}