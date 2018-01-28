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

void Classifier::GetTrainingData()
{
    // class list for training data
    vector<String> classList = GetFolderList(folderName);
    dirent *dirp;
    Mat img;
    GLCM glcm;
    svm_node* nodeList;
    for(String binClass : classList)
    {
        DIR *dir = opendir((folderName + binClass).c_str());
        vector<String> fileList;
        int cnt = 1;
        while((dirp = readdir(dir)) != nullptr)
        {
            if(dirp->d_type == DT_REG && dirp->d_name[0] != '.')
            {
                // calculate GLCM features
                glcm.Init(folderName + binClass + "/" + dirp->d_name, 16);
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

void Classifier::Train()
{
    svm_problem prob;
    // 训练样本的数目
    prob.l = (int)trainY.size();
    // 给x分配空间
    prob.x = &trainX[0];
    // 给y分配空间
    prob.y = &trainY[0];


    svm_parameter param;
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

    svm_model *model;
    model = svm_train(&prob, &param);
    svm_save_model((folderName + "svm_model").c_str(), model);
}



int Classifier::Predict(String path)
{
    GLCM glcm;
    svm_node* x;
    svm_model* model = svm_load_model((folderName + "svm_model").c_str());
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
