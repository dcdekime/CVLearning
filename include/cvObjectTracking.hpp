#include "visionCommon.hpp"

class cvObjectTracking
{
    // Lucas Kanade -> sparse optical flow: track only select points from frame to frame
    // Gunner Farneback -> dense optical flow: track all points from frame to frame
public:
    void sparseOpticalFlow();

};