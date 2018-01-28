//
// Created by Chen on 2018/1/27.
//

#include "GLCM.h"

GLCM::GLCM()
{

}

GLCM::GLCM(InputArray _srcImg, int _GLCMClass)
{
    srcImg = _srcImg.getMat();
    minMaxIdx(srcImg, NULL, &maxPixVal);
    if(_GLCMClass == -1)
        GLCMClass = (int)maxPixVal + 1;
    else
        GLCMClass = _GLCMClass;
    GLCMMat = *(new Mat(Size(GLCMClass, GLCMClass), CV_32FC1));
}

GLCM::GLCM(String path, int _GLCMClass)
{
    srcImg = imread(path, CV_8UC1);
    minMaxIdx(srcImg, NULL, &maxPixVal);
    if(_GLCMClass == -1)
        GLCMClass = (int)maxPixVal + 1;
    else
        GLCMClass = _GLCMClass;
    GLCMMat = *(new Mat(Size(GLCMClass, GLCMClass), CV_32FC1));
}

void GLCM::Init(String path, int _GLCMClass)
{
    srcImg = imread(path, CV_8UC1);
    minMaxIdx(srcImg, NULL, &maxPixVal);
    if(_GLCMClass == -1)
        GLCMClass = (int)maxPixVal + 1;
    else
        GLCMClass = _GLCMClass;
    GLCMMat = *(new Mat(Size(GLCMClass, GLCMClass), CV_32FC1));
}

GLCM::~GLCM()
{

}

/*
 * Brief:
 *      The function can calculate the GLCM matrix by using origin gray image
 *
 * Params:
 *      angle:      The angle for scanning, can be 0, 45, 90, 135
 *      offset:     The stride(offset) of the scanning
 *      norm:       Using normalization or not
 */
void GLCM::CalGLCM(int angle, int offset, bool norm)
{
    Size srcSize = srcImg.size();
    // using matrix temp to store the pixel in GLCM_class-gray
    Mat temp(srcSize, CV_8UC1);
    GLCMMat = Scalar_<float>(0);
    // zip srcImg into temp
    for(int h = 0; h < srcSize.height; ++h)
    {
        for(int w = 0; w < srcSize.width; ++w)
        {
            temp.at<uchar>(h, w) = (uchar)(srcImg.at<uchar>(h, w) * GLCMClass / (maxPixVal+1));
        }
    }
    // calculate the matrix
    int row = 0, col = 0;
    if(angle == 0)
    {
        for(int h = 0; h < srcSize.height; ++h)
        {
            uchar* tempLine = temp.ptr(h);
            for(int w = 0; w < srcSize.width - offset; ++w)
            {
                row = tempLine[w];
                col = tempLine[w + offset];
                GLCMMat.at<float>(row, col)++;
                GLCMMat.at<float>(col, row)++;
            }
        }
    }
    else if(angle == 90)
    {
        for(int h = 0; h < srcSize.height - offset; ++h)
        {
            uchar* tempLine = temp.ptr(h);
            uchar* tempLineOffset = temp.ptr(h + offset);
            for(int w = 0; w < srcSize.width; ++w)
            {
                row = tempLine[w];
                col = tempLineOffset[w];
                GLCMMat.at<float>(row, col)++;
                GLCMMat.at<float>(col, row)++;
            }
        }
    }
    else if(angle == 45)
    {
        for(int h = 0; h < srcSize.height - offset; ++h)
        {
            uchar* tempLine = temp.ptr(h);
            uchar* tempLineOffset = temp.ptr(h + offset);
            for(int w = 0; w < srcSize.width - offset; ++w)
            {
                row = tempLine[w];
                col = tempLineOffset[w + offset];
                GLCMMat.at<float>(row, col)++;
                GLCMMat.at<float>(col, row)++;
            }
        }
    }
    else if(angle == 135)
    {
        for(int h = 0; h < srcSize.height-offset; ++h)
        {
            uchar* tempLine = temp.ptr(h);
            uchar* tempLineOffset = temp.ptr(h + offset);
            for(int w = 1; w < srcSize.width; ++w)
            {
                row = tempLine[w];
                col = tempLineOffset[w - offset];
                GLCMMat.at<float>(row, col)++;
                GLCMMat.at<float>(col, row)++;
            }
        }
    }

    // normalization
    if(norm)
    {
        float sum = 0;
        for(int i = 0; i < GLCMClass; ++i)
        {
            for(int j = 0; j < GLCMClass; ++j)
            {
                sum += GLCMMat.at<float>(i, j);
            }
        }
        for(int i = 0; i < GLCMClass; ++i)
        {
            for(int j = 0; j < GLCMClass; ++j)
            {
                GLCMMat.at<float>(i, j) /= (sum * 1.0);
            }
        }
    }
}

/*
 * Brief:
 *      Using computed GLCMMat for caulating GLCMFeature
 */
void GLCM::CalFeature()
{
    GLCMFeature.entropy = 0;
    GLCMFeature.homogeneity = 0;
    GLCMFeature.contrast = 0;
    GLCMFeature.ASM = 0;
    GLCMFeature.correlation = 0;
    Size size = GLCMMat.size();
    float currVal = 0;
    // correlation
    double mean_i = 0, mean_j = 0, var_i = 0, var_j = 0;
    for(int i = 0; i < size.height; ++i)
    {
        float* GLCMLine = GLCMMat.ptr<float>(i);
        for(int j = 0; j < size.width; ++j)
        {
            currVal = GLCMLine[j];
            mean_i += currVal * i;
            mean_j += currVal * j;
            var_i += currVal * pow((i - mean_i), 2);
            var_j += currVal * pow((j - mean_j), 2);
        }
    }
    for(int h = 0; h < size.height; ++h)
    {
        float* GLCMLine = GLCMMat.ptr<float>(h);
        for(int w = 0; w < size.width; ++w)
        {
            currVal = GLCMLine[w];
            // Entropy
            if(currVal > 0)
                GLCMFeature.entropy += (pow(h-w, 2) * currVal) / log(currVal);
            // Contrast
            GLCMFeature.contrast += currVal * pow(h-w, 2);
            // Homogeneity
            GLCMFeature.homogeneity += currVal / (1+pow(h-w, 2));
            // Angular Second Moment
            GLCMFeature.ASM += pow(currVal, 2);
            // correlation
            GLCMFeature.correlation += currVal * (h-mean_i) * (w-mean_j);
        }
    }
    GLCMFeature.correlation /= sqrt(var_i*var_j);
}