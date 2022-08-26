#include "visionCommon.hpp"

class cvImageProcessing
{

public:
    void blendImages(cv::Mat& img1, cv::Mat& img2);
    void imageThresholding();
    void blurrSmooth();
    void morphological();
    void gradients();
    void histogram();
};