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
                glcm.Init(folderName + "/train/" + binClass + "/" + dirp->d_name, 16);
                glcm.CalGLCM(90);
                glcm.CalFeature();
                // add svm_node into trainX
                nodeList = new svm_node[glcm.GLCMFeature.featureNum+1];
                nodeList[0].index = 1; nodeList[0].value = glcm.GLCMFeature.entropy;
                nodeList[1].index = 2; nodeList[1].value = glcm.GLCMFeature.homogeneity;
                nodeList[2].index = 3; nodeList[2].value = glcm.GLCMFeature.contrast;
                nodeList[3].index = 4; nodeList[3].value = glcm.GLCMFeature.ASM;
                nodeList[4].index = 5; nodeList[4].value = glcm.GLCMFeature.correlation;
                nodeList[5].index = -1; nodeList[5].value = 0;
                trainX.push_back(nodeList);
                // add class num into trainY;
                trainY.push_back(stoi(binClass));
//                cout << dirp->d_name << " " << stoi(binClass) << endl;
//                cout << glcm.GLCMFeature.entropy << " " <<
//                        glcm.GLCMFeature.homogeneity << " " <<
//                        glcm.GLCMFeature.contrast << " " <<
//                        glcm.GLCMFeature.ASM << " " <<
//                        glcm.GLCMFeature.correlation << endl;
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
    // calculate GLCM features
    glcm.Init(path, 16);
    glcm.CalGLCM(90);
    glcm.CalFeature();
    // add svm_node into x
    x = new svm_node[glcm.GLCMFeature.featureNum+1];
    x[0].index = 1; x[0].value = glcm.GLCMFeature.entropy;
    x[1].index = 2; x[1].value = glcm.GLCMFeature.homogeneity;
    x[2].index = 3; x[2].value = glcm.GLCMFeature.contrast;
    x[3].index = 4; x[3].value = glcm.GLCMFeature.ASM;
    x[4].index = 5; x[4].value = glcm.GLCMFeature.correlation;
    x[5].index = -1; x[5].value = 0;
    return int(svm_predict(model, x));
}


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
    double label;
    bool findFlaw = false;
    for(int w=0; w<img.cols-15; w++)
    {
        for(int h=0; h<img.rows-15; h++)
        {
            Rect rect(w, h, 15, 15);
            subImg = img(rect);
            // calculate GLCM features
            glcm.Init(subImg, 16);
            glcm.CalGLCM(90);
            glcm.CalFeature();
            // add svm_node into x
            x = new svm_node[glcm.GLCMFeature.featureNum+1];
            x[0].index = 1; x[0].value = glcm.GLCMFeature.entropy;
            x[1].index = 2; x[1].value = glcm.GLCMFeature.homogeneity;
            x[2].index = 3; x[2].value = glcm.GLCMFeature.contrast;
            x[3].index = 4; x[3].value = glcm.GLCMFeature.ASM;
            x[4].index = 5; x[4].value = glcm.GLCMFeature.correlation;
            x[5].index = -1; x[5].value = 0;
            label = svm_predict(model, x);
            cout << w << ", " << h << ": " << label << endl;
            if(label == -1)
            {
                rectangle(rst, cvPoint(w, h), cvPoint(w+15, h+15), Scalar(0,0,255), 1, 1, 0);
                findFlaw = true;
                break;
            }
        }
        if(findFlaw)
        {
            break;
        }
    }
    imwrite(rstPath, rst);
}


void Classifier::ProcessImgByCover(String srcPath, String rstPath)
{
    // src
    Mat src = imread(srcPath, CV_8UC1);
    Size size = src.size();
    // srcFront
    Mat srcFront, temp1, temp2, srcFrontCopy;
    threshold(src, temp1, 0, 255, CV_THRESH_OTSU);
    dilate(temp1, temp2, getStructuringElement(MORPH_RECT, Size(3, 3)));
    erode(temp2, srcFront, getStructuringElement(MORPH_RECT, Size(11, 11)));
//    srcFront.copyTo(srcFrontCopy);
    // integral image
    Mat srcIntImg;
    Mat frontIntImg;
    // rst
    Mat rst;
    src.copyTo(rst);

    Mat subImg;
    GLCM glcm;
    svm_node* x;
    svm_model* model = svm_load_model((folderName + "/svm_model").c_str());
    double label;
    bool findFlaw = false;

    int x1, y1, x2, y2;
    int count;
    double sum, frontSum;
    int pad = (15 - 1) / 2;
    GetIntegralImage(src, srcIntImg);
    GetIntegralImage(srcFront, frontIntImg);
    for(int h = 0; h < size.height-pad; h+=15)
    {
        for(int w = 0; w < size.width-pad; w+=15)
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
                // calculate GLCM features
                glcm.Init(subImg, 16);
                glcm.CalGLCM(90);
                glcm.CalFeature();
                // add svm_node into x
                x = new svm_node[glcm.GLCMFeature.featureNum+1];
                x[0].index = 1; x[0].value = glcm.GLCMFeature.entropy;
                x[1].index = 2; x[1].value = glcm.GLCMFeature.homogeneity;
                x[2].index = 3; x[2].value = glcm.GLCMFeature.contrast;
                x[3].index = 4; x[3].value = glcm.GLCMFeature.ASM;
                x[4].index = 5; x[4].value = glcm.GLCMFeature.correlation;
                x[5].index = -1; x[5].value = 0;
                label = svm_predict(model, x);
//                cout << h << ", " << w << ": " << label << endl;
                if(label == -1)
                {
                    rectangle(rst, cvPoint(x1, y1), cvPoint(x2, y2), Scalar(0,0,255), 1, 1, 0);
//                    rectangle(srcFrontCopy, cvPoint(w, h), cvPoint(w+15, h+15), Scalar(0,0,255), 1, 1, 0);
                    findFlaw = true;
                    break;
                }
            }
        }
//        if(findFlaw)
//        {
//            break;
//        }
    }
//    imwrite("../binary_classification/test/10_temp.jpg", srcFrontCopy);
    imwrite(rstPath, rst);
}
