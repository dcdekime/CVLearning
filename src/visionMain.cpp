#include "cvFunctions.hpp"
#include "cvImageProcessing.hpp"

int main( int argc, char** argv )
{
    cv::Mat img1 = cv::imread("/home/parallels/Documents/Vision_Stuff/resources/pup1.jpg",
            cv::IMREAD_COLOR);

    cv::Mat img2 = cv::imread("/home/parallels/Documents/Vision_Stuff/resources/pup2.jpg",
            cv::IMREAD_COLOR);

    cv::Mat phtevenImg = cv::imread("/home/parallels/Documents/Vision_Stuff/resources/phteven.jpg",
            cv::IMREAD_COLOR);

    cvFunctions cvObj;
    cvImageProcessing ipObj;

    // *** call cvFunctions methods ***
    //cvObj.importImage();
    cvObj.captureWebcamVideo();
    //cvObj.drawOnImage();

    // *** call imageProcessing methods ***
    //ipObj.blendImages(img1, phtevenImg);
    //ipObj.imageThresholding();
    //ipObj.blurrSmooth();
    //ipObj.morphological();
    //ipObj.gradients();
    //ipObj.histogram();

    return 0;
}