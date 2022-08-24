#include <iostream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"

class cvObjectDetection
{

public:
    void templateMatching();
    void cornerDetection();
    void edgeDetection();
    void contourDetection();
    void featureDetection(const std::string& fmType);
    void segmentationNoWatershed();
    void watershedSegmentation();
    void watershedCustomSeed();
};