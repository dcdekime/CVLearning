#include "cvFunctions.hpp"
#include "cvImageProcessing.hpp"
#include "cvObjectDetection.hpp"
#include "cvObjectTracking.hpp"


int main(int argc, char** argv)
{
        std::string pup1 = "pup1.jpg";
        cv::Mat img1 = cv::imread(RESOURCE_PATH + pup1, cv::IMREAD_COLOR);

        std::string pup2 = "pup2.jpg";
        cv::Mat img2 = cv::imread(RESOURCE_PATH + pup2, cv::IMREAD_COLOR);

        std::string phteven = "phteven.jpg";
        cv::Mat phtevenImg = cv::imread(RESOURCE_PATH + phteven, cv::IMREAD_COLOR);

        cvFunctions cvObj;
        cvImageProcessing ipObj;
        cvObjectDetection objDet;
        cvObjectTracking objTrack;

        // *** call cvFunctions methods ***
        //cvObj.importImage();
        //cvObj.captureWebcamVideo();
        //cvObj.drawOnImage();

        // *** call imageProcessing methods ***
        //ipObj.blendImages(img1, phtevenImg);
        //ipObj.imageThresholding();
        //ipObj.blurrSmooth();
        //ipObj.morphological();
        //ipObj.gradients();
        //ipObj.histogram();

        // *** call objectDetection methods ***
        std::string featureDetectionType = "SIFT";
        //objDet.templateMatching();
        //objDet.cornerDetection();
        //objDet.edgeDetection();
        //objDet.contourDetection();
        //objDet.featureDetection(featureDetectionType);
        //objDet.watershedSegmentation();
        //objDet.faceDetection();
        //objDet.blurPlates();

        // *** call objectTracking methods ***
        //objTrack.sparseOpticalFlow();
    
        return 0;
}