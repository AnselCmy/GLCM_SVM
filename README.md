![Build Status](https://img.shields.io/travis/USER/REPO.svg)
![opencv](https://img.shields.io/badge/opencv-v3.3.1-blue.svg)  

This is a tool for texture image classification by using GLCM and SVM algorithm.

## Example
You need to put your training data into ../binary_classification/train, and put the negative texture images into folder -1, and positive images into folder 1, just like the example folder tree here.  

```
├── binary_classification
    └── train
        ├── -1 (**your negative training images here)
        └── 1  (**your negative training images here)
```
In your main.cpp

```c++
#include "GLCM.h"
#include "Classifier.h"

int main()
{
    Classifier classifier("../binary_classification");
    classifier.GetTrainingData();
    classifier.Train();
    return 0;
}
```
After running such main function, you will have a file named svm_model under the folder binary_classification 

```
├── binary_classification
    ├── svm_model (**here is your model by training your data)
    └── train
        ├── -1 
        └── 1 
```

## Folder Tree
```
├── CMakeLists.txt
├── Classifier.cpp
├── Classifier.h
├── GLCM.cpp
├── GLCM.h
├── binary_classification
│   ├── svm_model
│   └── train
│       ├── -1
│       └── 1
├── main.cpp
├── svm.cpp
└── svm.h
```