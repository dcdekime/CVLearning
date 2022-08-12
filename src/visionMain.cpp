#include "cvFunctions.hpp"
#include "cvImageProcessing.hpp"
#include "cvObjectDetection.hpp"

int main( int argc, char** argv )
{
        cv::Mat img1 = cv::imread("/home/ddekime/CVLearning/resources/pup1.jpg",
            cv::IMREAD_COLOR);

        cv::Mat img2 = cv::imread("/home/ddekime/CVLearning/resources/pup2.jpg",
            cv::IMREAD_COLOR);

        cv::Mat phtevenImg = cv::imread("/home/ddekime/CVLearning/resources/phteven.jpg",
                cv::IMREAD_COLOR);

        cvFunctions cvObj;
        cvImageProcessing ipObj;
        cvObjectDetection objDet;

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
        objDet.featureDetection(featureDetectionType);
    
        return 0;
}