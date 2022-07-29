#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"

class cvFunctions
{
public:
    cvFunctions();
    cvFunctions(cv::Mat& img);

    void importImage();
    void captureWebcamVideo();
    void drawOnImage();

private:

    cv::Mat mImg;
};