cmake_minimum_required(VERSION 3.9)
project(GLCM_SVM)
find_package(OpenCV3)
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp GLCM.cpp GLCM.h)
add_executable(GLCM_SVM ${SOURCE_FILES})
find_package(OpenCV REQUIRED)
target_link_libraries(GLCM_SVM ${OpenCV_LIBS})