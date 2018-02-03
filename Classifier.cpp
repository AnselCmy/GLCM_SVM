//
// Created by Chen on 2018/1/28.
//

#include "Classifier.h"

using namespace std;
using namespace cv;


Classifier::Classifier(String _folderName)
{
    folderName = _folderName;
}


Classifier::~Classifier()
{

}


/*
 * Brief:
 *      Get the list of folders of the input folderName
 *
 * Param:
 *      The name of the folder need to be listed
 */
vector<String> Classifier::GetFolderList(String folderName)
{
    DIR *dir = opendir(folderName.c_str());
    dirent *dirp;
    vector<String> folderList;
    while((dirp = readdir(dir)) != nullptr)
    {
        if(dirp->d_type == DT_DIR && dirp->d_name[0] != '.')
        {
            folderList.push_back(dirp->d_name);
        }
    }
    closedir(dir);
    return folderList;
}


/*
 * Brief:
 *      Transform the data from [folderName]/train to get trainX and trainY by GLCM
 */
void Classifier::GetTrainingData()
{
    // class list for training data
    vector<String> classList = GetFolderList(folderName + "/train");
    dirent *dirp;
    Mat img;
    GLCM glcm;
    svm_node* nodeList;
    vector<double> features;
    for(String binClass : classList)
    {
        DIR *dir = opendir((folderName + "/train/" + binClass).c_str());
        vector<String> fileList;
        int cnt = 1;
        while((dirp = readdir(dir)) != nullptr)
        {
            if(dirp->d_type == DT_REG && dirp->d_name[0] != '.')
            {
                // calculate GLCM features
                int angleList[] = {0, 45, 90, 135};
                glcm.Init(folderName + "/train/" + binClass + "/" + dirp->d_name, 16);
                features = glcm.GetFeaturesByAngle(angleList, 4);
                // add svm_node into trainX
                nodeList = new svm_node[features.size()+1];
                for(int i=0; i<features.size(); i++)
                {
                    nodeList[i].index = i+1;
                    nodeList[i].value = features[i];
                }
                nodeList[features.size()].index = -1;
                nodeList[features.size()].value = 0;
                trainX.push_back(nodeList);
                // add class num into trainY;
                trainY.push_back(stoi(binClass));
//                cout << dirp->d_name << " " << stoi(binClass) << endl;
//                cout << glcm.GLCMFeature.entropy << " " <<
//                        glcm.GLCMFeature.homogeneity << " " <<
//                        glcm.GLCMFeature.contrast << " " <<
//                        glcm.GLCMFeature.ASM << " " <<
//                        glcm.GLCMFeature.correaltion << endl;
//                cout << "--------------------------------------" << endl;
            }
        }
        closedir(dir);
    }
}


/*
 * Brief:
 *      Initialize the svm_problem by trainX and trainY
 */
void Classifier::InitProb()
{
    // 训练样本的数目
    prob.l = (int)trainY.size();
    // 给x分配空间
    prob.x = &trainX[0];
    // 给y分配空间
    prob.y = &trainY[0];
}


/*
 * Brief:
 *      Initialize the svm_parameter
 */
void Classifier::InitParam()
{
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.gamma = 0.5;	// 1/num_features
    param.cache_size = 100;
    param.C = 2;
    param.eps = 1e-3;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
}


/*
 * Brief:
 *      Using the prob and param to do cross validation
 *
 * Param:
 *      foldNum: The number of fold in cross validation
 */
double Classifier::CrossValidation(int foldNum)
{
    double* target = new double[prob.l];
    int correct = 0;
    svm_cross_validation(&prob, &param, foldNum, target);
    for(int i = 0; i < prob.l; i++)
        if(target[i] == prob.y[i])
            correct++;
    return 1.0*correct/prob.l;
}


/*
 * Brief:
 *      Using the prob and param to train the svm_model
 */
void Classifier::Train()
{
    svm_model *model;
    model = svm_train(&prob, &param);
    svm_save_model((folderName + "/svm_model").c_str(), model);
}


/*
 * Brief:
 *      Using the path of the image to predict its class
 *
 * Param:
 *      path: The path of image need to be predicted
 *
 * Return:
 *      The class of the input image
 */
int Classifier::Predict(String path)
{
    GLCM glcm;
    svm_node* x;
    svm_model* model = svm_load_model((folderName + "/svm_model").c_str());
    vector<double> features;
    // calculate GLCM features
    glcm.Init(path, 16);
    int angleList[] = {0, 45, 90, 135};
    features = glcm.GetFeaturesByAngle(angleList, 4);
    // add svm_node into x
    x = new svm_node[features.size()+1];
    for(int i=0; i<features.size(); i++)
    {
        x[i].index = i+1;
        x[i].value = features[i];
    }
    x[features.size()].index = -1;
    x[features.size()].value = 0;
    return int(svm_predict(model, x));
}


/*
 * Brief:
 *      Using trained model and img Mat to predict its class
 *
 * Param:
 *      model:  Trained svm model
 *      img:    Mat of a image
 *
 * Return:
 *      The class for this image
*/
int Classifier::Predict(svm_model* model, const Mat img)
{
    GLCM glcm;
    svm_node* x;
    vector<double> features;
    // calculate GLCM features
    glcm.Init(img, 16);
    int angleList[] = {0, 45, 90, 135};
    features = glcm.GetFeaturesByAngle(angleList, 4);
    // add svm_node into x
    x = new svm_node[features.size()+1];
    for(int i=0; i<features.size(); i++)
    {
        x[i].index = i+1;
        x[i].value = features[i];
    }
    x[features.size()].index = -1;
    x[features.size()].value = 0;
    return int(svm_predict(model, x));
}


/*
 * Brief:
 *      Get the integral image for _src image
 *
 * Param:
 *      _src:   The source image
 *      _intImg:The integral image for src image
 *      power:  The power for every pixel, usually 1
*/
void Classifier::GetIntegralImage(InputArray _src, OutputArray _intImg, int power)
{
    // src
    Mat src = _src.getMat();
    Size size = src.size();
    // integral image
    _intImg.create(size, CV_64FC1);
    Mat intImg = _intImg.getMat();

    double sum;
    for(int w = 0; w < size.width; ++w)
    {
        sum = 0;
        for(int h = 0; h < size.height; ++h)
        {
            if(power == 1)
                sum += src.at<uchar>(h, w);
            else
                sum += pow(src.at<uchar>(h, w), power);
            if(w == 0)
                intImg.at<double>(h, w) = sum;
            else
                intImg.at<double>(h, w) = intImg.at<double>(h, w-1) + sum;
        }
    }
}


void Classifier::ProcessImg(String srcPath, String rstPath)
{
    Mat img = imread(srcPath, CV_8UC1);
    Mat rst;
    img.copyTo(rst);
    Mat subImg;
    GLCM glcm;
    svm_node* x;
    svm_model* model = svm_load_model((folderName + "/svm_model").c_str());
    vector<double> features;
    double label;
    bool findFlaw = false;
    for(int w=0; w<img.cols-15; w++)
    {
        for(int h=0; h<img.rows-15; h++)
        {
            Rect rect(w, h, 15, 15);
            subImg = img(rect);
            label = Predict(model, subImg);
            if(label == -1)
            {
                rectangle(rst, cvPoint(w, h), cvPoint(w+15, h+15), Scalar(0,0,255), 1, 1, 0);
                findFlaw = true;
            }
        }
    }
    imwrite(rstPath, rst);
}


void Classifier::ProcessImgByCover(String srcPath, String rstPath, String tempPath)
{
    // src
//    Mat srctemp = imread(srcPath, CV_8UC1);
//    Mat src;
//    GaussianBlur(srctemp, src, Size(5, 5), 0, 0);
    Mat src = imread(srcPath, CV_8UC1);
    Size size = src.size();
    // srcFront
    Mat srcFront, temp1, temp2, srcFrontCopy, srcFrontCopy2;
//    threshold(src, srcFrontCopy, 0, 255, CV_THRESH_OTSU);
    threshold(src, srcFront, 0, 255, CV_THRESH_OTSU);
//    dilate(temp1, temp2, getStructuringElement(MORPH_RECT, Size(3, 3)));
//    erode(temp2, srcFront, getStructuringElement(MORPH_RECT, Size(3, 3)));
    srcFront.copyTo(srcFrontCopy);
    srcFront.copyTo(srcFrontCopy2);
    // integral image
    Mat srcIntImg;
    Mat frontIntImg;
    // rst
    Mat rstTemp, rst;
    src.copyTo(rstTemp);
    cvtColor(rstTemp, rst, COLOR_GRAY2BGR);

    Mat subImg;
    GLCM glcm;
    svm_node* x;
    svm_model* model = svm_load_model((folderName + "/svm_model").c_str());
    vector<double> features;
    double label;
    bool findFlaw = false;

    int x1, y1, x2, y2;
    int count;
    double sum, frontSum;
    int pad = (15 - 1) / 2;
    GetIntegralImage(src, srcIntImg);
    GetIntegralImage(srcFront, frontIntImg);
    for(int h = 0; h < size.height-pad; h+=5)
    {
        for(int w = 0; w < size.width-pad; w+=5)
        {
            if(srcFront.at<uchar>(h, w) == 255)
            {
                x1 = w;
                x2 = w + 14;
                y1 = h;
                y2 = h + 14;

                frontSum = frontIntImg.at<double>(y2, x2) + frontIntImg.at<double>(y1 - 1, x1 - 1)
                           - frontIntImg.at<double>(y1 - 1, x2) - frontIntImg.at<double>(y2, x1 - 1);
                if(frontSum != 255 * 15 * 15)
                {
                    continue;
                }

                Rect rect(x1, y1, 15, 15);
                subImg = src(rect);
//                rectangle(srcFrontCopy2, cvPoint(x1, y1), cvPoint(x2, y2), Scalar(0,0,100), 1, 1, 0);
//                Mat subImgGauss;
//                GaussianBlur(subImg, subImgGauss, Size(5, 5), 0, 0);
                label = Predict(model, subImg);
//                cout << h << ", " << w << ": " << label << endl;
                if(label == -1)
                {
                    rectangle(rst, cvPoint(x1-1, y1-1), cvPoint(x2+1, y2+1), Scalar(0,0,255), 1, 1, 0);
                    rectangle(srcFrontCopy, cvPoint(x1-1, y1-1), cvPoint(x2+1, y2+1), Scalar(0,0,255), 1, 1, 0);
                    findFlaw = true;
//                    break;
                }
            }
        }
//        if(findFlaw)
//        {
//            break;
//        }
    }
    if(tempPath != "")
        imwrite(tempPath, srcFrontCopy);
    if(rstPath != "")
        imwrite(rstPath, rst);
}
