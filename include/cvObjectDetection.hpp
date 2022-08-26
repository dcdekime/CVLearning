#include "visionCommon.hpp"

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
    void faceDetection();
    void blurPlates();
};