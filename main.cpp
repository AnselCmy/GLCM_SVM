#include <iostream>
#include <opencv2/opencv.hpp>
#include "GLCM.h"
#include "Classifier.h"

using namespace std;
using namespace cv;

int main()
{
//    Mat src = imread("../binary_classification/test/34.bmp", CV_8UC1);
//    Mat temp1, temp2, srcFront;
//    threshold(src, temp1, 0, 255, CV_THRESH_OTSU);
//    dilate(temp1, temp2, getStructuringElement(MORPH_RECT, Size(3, 3)));
//    erode(temp2, srcFront, getStructuringElement(MORPH_RECT, Size(3, 3)));
//    imwrite("../binary_classification/test/34_dilate_erode.bmp", srcFront);

    Classifier classifier("../binary_classification");
//    Mat srctemp, src;
//    srctemp = imread("../binary_classification/test/12.bmp", CV_8UC1);
//    GaussianBlur(srctemp, src, Size(5, 5), 0, 0);
//    imwrite("../binary_classification/test/12_gauss.jpg", src);

//    classifier.GetTrainingData();
//    classifier.InitProb();
//    classifier.InitParam();
//    cout << classifier.CrossValidation(3) << endl;
//    classifier.Train();
//    GLCM glcm;
//    glcm.Init("../binary_classification/train/-1/1209.jpg", 16);
//    glcm.CalGLCM();
//    cout << glcm.srcImg << endl;
//    cout << glcm.GLCMMat << endl;
//    glcm.CalFeature();
//    cout << glcm.GLCMFeature.entropy << endl;
//    cout << glcm.GLCMFeature.homogeneity << endl;
//    cout << glcm.GLCMFeature.contrast << endl;
//    cout << glcm.GLCMFeature.ASM << endl;
//    cout << glcm.GLCMFeature.correlation << endl;

//    cout << classifier.Predict("../wood_grass/train/1/2000.jpg") << endl;

//    svm_model* model = svm_load_model((classifier.folderName + "/svm_model").c_str());
//    Mat img = imread("../binary_classification/test/23.bmp", CV_8UC1);
//    cout << classifier.Predict(model, img);

//    -0.478236 0.609048 1.20476 0.0616327 0.100664 -0.502702 0.6 1.26531 0.0622137 0.0993249 -0.0401525 0.930952 0.138095 0.137903 0.125585 -0.46334
//    for(int i=0; i<2000; i++)
//    {
//        String folder = "../binary_classification/train/1/";
//        cout << folder+to_string(i)+".png" << endl;
//        cout << classifier.Predict(folder+to_string(i)+".png") << endl;
//    }
//    cout << classifier.Predict("../binary_classification/test/33.png") << endl;

//    classifier.ProcessImg("../binary_classification/test/8.bmp", "../binary_classification/test/8_rst.png");
    classifier.ProcessImgByCover("../binary_classification/test/13.bmp",
                                 "../binary_classification/test/13_rst.png",
                                 "../binary_classification/test/13_temp.png");

//    svm_model* model = svm_load_model((classifier.folderName + "/svm_model").c_str());
//    Mat img = imread("../binary_classification/test/22.bmp", CV_8UC1);
//    Mat imgGauss;
//    GaussianBlur(img, imgGauss, Size(5, 5), 0, 0);
//    cout << classifier.Predict(model, imgGauss) << endl;
    return 0;
}