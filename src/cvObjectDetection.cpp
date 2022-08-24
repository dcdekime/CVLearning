#include "cvObjectDetection.hpp"

#define RESOURCE_PATH "/home/ddekime/CVLearning/resources/"

void cvObjectDetection::templateMatching()
{
    cv::Mat sammyFull = cv::imread("/home/ddekime/CVLearning/resources/sammy.jpg");
    cv::Mat sammyFace = cv::imread("/home/ddekime/CVLearning/resources/sammy_face.jpg");
    cv::Mat similarityMap, sammyFullCopy;
    sammyFull.copyTo(sammyFullCopy);

    // *** Do template matching ***
    /*
    Feature matching methods mapping
    0: cv::TM_CCOEFF
    1: cv::TM_CCOEFF_NORMED
    2: cv::TM_CCORR
    3: cv::TM_CCORR_NORMED
    4: cv::TM_CCORR_NORMED
    5: cv::TM_SQDIFF
    6: cv::TM_SQDIFF_NORMED
    */
    int matchMethod = 3;
    cv::matchTemplate(sammyFullCopy, sammyFace, similarityMap, matchMethod);

    // *** Draw rectangle around template match region
    double minVal, maxVal;
    cv::Point minPt, maxPt, matchLocationStart;
    cv::minMaxLoc(similarityMap, &minVal, &maxVal, &minPt, &maxPt);

    if (matchMethod == cv::TM_SQDIFF || matchMethod == cv::TM_SQDIFF_NORMED)
        matchLocationStart = minPt;
    else
        matchLocationStart = maxPt;
    
    cv::rectangle(sammyFullCopy, 
        matchLocationStart, 
        cv::Point(matchLocationStart.x + sammyFace.cols, matchLocationStart.y + sammyFace.rows),
        cv::Scalar(0,0,255),
        2, 8, 0);


    std::string heatMapWin = "heatMap";
    std::string fullWinName = "sammy_full";
    cv::namedWindow(fullWinName);
    cv::namedWindow(heatMapWin);


    cv::imshow(fullWinName, sammyFullCopy);
    cv::imshow(heatMapWin, similarityMap);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void cvObjectDetection::cornerDetection()
{
    // Read in chessboard images
    cv::Mat flatChess = cv::imread("/home/ddekime/CVLearning/resources/flat_chessboard.png", cv::IMREAD_COLOR);
    cv::Mat flatChessGray;
    cv::cvtColor(flatChess, flatChessGray, cv::COLOR_BGR2GRAY);
    
    cv::Mat realChess = cv::imread("/home/ddekime/CVLearning/resources/real_chessboard.jpg", cv::IMREAD_COLOR);
    cv::Mat realChessGray = cv::imread("/home/ddekime/CVLearning/resources/real_chessboard.jpg", CV_32FC1);
    realChessGray.convertTo(realChessGray, CV_32FC1);

    // Perform Harris Corner Detection
    cv::Mat flatChessCorners = cv::Mat::zeros(flatChessGray.size(), CV_32FC1);
    cv::cornerHarris(flatChessGray, flatChessCorners, 2, 3, 0.04);
    
    cv::Mat flatChessCorners_norm, flatChessCorners_norm_scaled;
    cv::normalize( flatChessCorners, flatChessCorners_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
    // std::cout << "flat chess corners norm:" << std::endl;
    // std::cout << flatChessCorners_norm << std::endl;
    convertScaleAbs(flatChessCorners_norm, flatChessCorners_norm_scaled );
    int thresh = 200;

    for( int i = 0; i < flatChessCorners_norm.rows ; i++ )
    {
        for( int j = 0; j < flatChessCorners_norm.cols; j++ )
        {
            if( (int) flatChessCorners_norm.at<float>(i,j) > thresh )
            {
                cv::circle(flatChessCorners_norm_scaled, cv::Point(j,i), 5,  cv::Scalar(0), 2, 8, 0 );
            }
        }
    }

    cv::namedWindow("flatChessGray");
    cv::namedWindow("chessCorners");
    cv::imshow("chessCorners", flatChessCorners_norm_scaled);
    cv::imshow("flatChessGray", flatChessGray);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // *** For improved corner detection, see Shi-Tomasi Detection! -> "GoodFeaturesToTrack" ***
}

// initialize matrices and parameters
cv::Mat src, srcGray;
cv::Mat dst, detectedEdges;

int lowThreshold = 0;
const int max_lowThreshold = 100;
const int ratio = 3;
const int kernel_size = 3;
const std::string windowName = "Edge Map";

void CannyEdgeDetector(int, void*)
{
    // De-noise image by blurring
    cv::blur(srcGray, detectedEdges, cv::Size(3,3));

    cv::Canny(detectedEdges, detectedEdges, lowThreshold, lowThreshold*ratio, kernel_size);

    dst = cv::Scalar::all(0);

    src.copyTo(dst, detectedEdges);

    // Display Output
    cv::imshow("Edge Map", dst);
}

void cvObjectDetection::edgeDetection()
{
    src = cv::imread("/home/ddekime/CVLearning/resources/sammy_face.jpg", cv::IMREAD_COLOR);
    dst.create(src.size(), src.type());
    cv::cvtColor(src, srcGray, cv::COLOR_BGR2GRAY);

    // Call Edge Detection
    cv::namedWindow("originalImage");
    cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);
    cv::createTrackbar("Min Threshold:", windowName, &lowThreshold, max_lowThreshold, CannyEdgeDetector);
    CannyEdgeDetector(0,0);

    cv::imshow("originalImage", src);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

cv::Mat src_gray;
int thresh = 100;
cv::RNG rng(12345);

void thresh_callback(int, void*)
{
    // use Canny Edge Detection
    cv::Mat cannyOutput;
    cv::Canny(src_gray, cannyOutput, thresh, thresh*2);

    // Find contours using edge detection output
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    // define what kind of features to extract -> e.g. external, internal, all
    cv::findContours(cannyOutput, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat drawing = cv::Mat::zeros(cannyOutput.size(), CV_8UC3);
    for (int i=0; i < contours.size(); i++)
    {
        cv::Scalar color = cv::Scalar(rng.uniform(0,256), rng.uniform(0,256), rng.uniform(0,256));
        cv::drawContours(drawing, contours, (int)i, color, 2, cv::LINE_8, hierarchy);
    }

    cv::imshow("Contours", drawing);
}

void cvObjectDetection::contourDetection()
{
    cv::Mat src =  cv::imread("/home/ddekime/CVLearning/resources/internal_external.png", cv::IMREAD_COLOR);
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
    blur(src_gray, src_gray, cv::Size(3,3));

    // Call Edge Detection
    int maxThresh = 255;
    const char* source_window = "Source";
    cv::namedWindow(source_window, CV_WINDOW_AUTOSIZE);
    cv::createTrackbar("Min Threshold:", source_window, &thresh, maxThresh, thresh_callback);
    thresh_callback(0,0);

    cv::namedWindow("originalImage");
    cv::imshow("originalImage", src);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void cvObjectDetection::featureDetection(const std::string& fmType)
{
    cv::Mat reeses = cv::imread("/home/ddekime/CVLearning/resources/reeses_puffs.png", cv::IMREAD_GRAYSCALE);
    cv::Mat cereals = cv::imread("/home/ddekime/CVLearning/resources/many_cereals.jpg", cv::IMREAD_GRAYSCALE);
    
    std::vector<cv::KeyPoint> keyPoints1, keyPoints2;
    std::vector<cv::DMatch> matches;
    cv::Mat descriptor1, descriptor2, matchImg;

    if (fmType == "ORB")
    {
        cv::Ptr<cv::ORB> orb = cv::ORB::create();
        orb->detectAndCompute(reeses, cv::Mat(), keyPoints1, descriptor1);
        orb->detectAndCompute(cereals, cv::Mat(), keyPoints2, descriptor2);

        cv::Ptr<cv::BFMatcher> bf = cv::BFMatcher::create(cv::NORM_HAMMING, true);
        bf->match(descriptor1, descriptor2, matches);
        
        std::sort(matches.begin(), matches.end(), [](cv::DMatch& m1, cv::DMatch& m2) {return m1.distance < m2.distance;});
        std::vector<cv::DMatch> topXMatches(matches.begin(), matches.begin()+26);

        cv::drawMatches(reeses, keyPoints1, cereals, keyPoints2, topXMatches, matchImg);
    }
    else if (fmType == "SIFT")
    {
        std::vector<std::vector<cv::DMatch>> matchVec;
        std::vector<cv::DMatch> goodMatches;

        cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
        sift->detectAndCompute(reeses, cv::Mat(), keyPoints1, descriptor1);
        sift->detectAndCompute(cereals, cv::Mat(), keyPoints2, descriptor2);

        cv::Ptr<cv::BFMatcher> bf = cv::BFMatcher::create();
        bf->knnMatch(descriptor1, descriptor2, matchVec, 2); // knn size = 2 -> creates vector of match pairs

        // LESS DISTANCE == BETTER MATCH
        // RATIO MATCH 1 < 75% MATCH 2
        for (const auto& match12 : matchVec)
        {
            // if match1 distance is less than 75% of match2 distance
            // then descriptor was a good match, lets keep it
            if (match12[0].distance < 0.75 * match12[1].distance)
            {
                goodMatches.push_back(match12[0]);
            }
        }

        cv::drawMatches(reeses, keyPoints1, cereals, keyPoints2, goodMatches, matchImg);
    }
    else if (fmType == "FLANN")
    {
        std::vector<std::vector<cv::DMatch>> matchVec;
        std::vector<cv::DMatch> goodMatches;

        cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
        sift->detectAndCompute(reeses, cv::Mat(), keyPoints1, descriptor1);
        sift->detectAndCompute(cereals, cv::Mat(), keyPoints2, descriptor2);

        cv::Ptr<cv::FlannBasedMatcher> flann =  cv::FlannBasedMatcher::create();
        flann->knnMatch(descriptor1, descriptor2, matchVec, 2);

        // LESS DISTANCE == BETTER MATCH
        // RATIO MATCH 1 < 75% MATCH 2
        for (const auto& match12 : matchVec)
        {
            // if match1 distance is less than 75% of match2 distance
            // then descriptor was a good match, lets keep it
            if (match12[0].distance < 0.75 * match12[1].distance)
            {
                goodMatches.push_back(match12[0]);
            }
        }

        cv::drawMatches(reeses, keyPoints1, cereals, keyPoints2, goodMatches, matchImg);
    }

    cv::namedWindow("reeses");
    cv::namedWindow("cereals");
    cv::namedWindow("matches");

    cv::imshow("reeses", reeses);
    cv::imshow("cereals", cereals);
    cv::imshow("matches", matchImg);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void cvObjectDetection::segmentationNoWatershed()
{
    std::string pennyImgName = "pennies.jpg";
    cv::Mat pennyImg = cv::imread(RESOURCE_PATH + pennyImgName, cv::IMREAD_COLOR);

    // Median Blur
    cv::Mat sepBlur, sepBlurGray, sepThresh;
    cv::medianBlur(pennyImg, sepBlur, 25);

    // GrayScale
    cv::cvtColor(sepBlur, sepBlurGray, cv::COLOR_BGR2GRAY);

    // Binary Threshold
    cv::threshold(sepBlurGray, sepThresh, 160, 255, cv::THRESH_BINARY_INV);

    // Find Contours

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    // define what kind of features to extract -> e.g. external, internal, all
    cv::findContours(sepThresh, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (int i=0; i < contours.size(); i++)
    {
        cv::Scalar color = cv::Scalar(0,0,255); // red contour (BGR)
        cv::drawContours(pennyImg, contours, (int)i, color, 10, cv::LINE_8, hierarchy);
    }

    // Display Contours Image
    std::string pennyWindow = "pennies";
    std::string contourWindow = "contours";
    cv::namedWindow(pennyWindow, cv::WindowFlags::WINDOW_NORMAL);
    cv::imshow(pennyWindow, pennyImg);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void cvObjectDetection::watershedSegmentation()
{
    std::string pennyImgName = "pennies.jpg";
    cv::Mat pennyImg = cv::imread(RESOURCE_PATH + pennyImgName, cv::IMREAD_COLOR);

    // Median Blur
    cv::Mat sepBlur, sepBlurGray, sepThresh;
    cv::medianBlur(pennyImg, sepBlur, 35);

    // GrayScale
    cv::cvtColor(sepBlur, sepBlurGray, cv::COLOR_BGR2GRAY);

    // Binary Threshold
    // Otsu's method -> cluster-based image thresholding algorithm
    cv::threshold(sepBlurGray, sepThresh, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);

    // Noise Removal
    cv::Mat kernel = cv::Mat::ones(cv::Size(3,3), CV_8UC1);
    cv::Mat regionBG;
    regionBG.convertTo(regionBG, CV_8U);
    cv::morphologyEx(sepThresh, regionBG, cv::MORPH_OPEN, kernel, cv::Point(-1,-1), 2);

    // Distance Transform
    cv::Mat distTransform;
    distTransform.convertTo(distTransform, CV_8U);
    cv::distanceTransform(regionBG, distTransform, cv::DIST_L2, 5);
    cv::normalize(distTransform, distTransform, 0, 1.0, cv::NORM_MINMAX);

    // Threshold (dilate) again to get sure points
    cv::Mat confRegionFG;
    double minVal, maxVal;
    cv::minMaxIdx(distTransform, &minVal, &maxVal);
    cv::threshold(distTransform, confRegionFG, 0.7*maxVal, 255, 0);

    // Find unknown region
    cv::Mat unknownRegion;
    confRegionFG.convertTo(confRegionFG, CV_8U);
    cv::subtract(regionBG, confRegionFG, unknownRegion);

    // Create label markers for watershed algorithm

    // 1. get markers
    cv::Mat markers = cv::Mat(confRegionFG.size(), CV_32S);
    cv::connectedComponents(confRegionFG, markers);
    markers += 1;
    cv::Mat backgroundMask = unknownRegion == 255;
    markers.setTo(0, backgroundMask);

    // perform watershed algorithm
    cv::watershed(pennyImg, markers);
    markers.convertTo(markers, CV_8U, 10);

    // Find Contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    // define what kind of features to extract -> e.g. external, internal, all
    cv::findContours(markers, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    for (int i=0; i < contours.size(); i++)
    {
        cv::Scalar color = cv::Scalar(255,0,0); // red contour (BGR)
        cv::drawContours(pennyImg, contours, (int)i, color, 10, cv::LINE_8, hierarchy);
    }


    // Display Contours Image
    std::string pennyWindow = "pennies";
    std::string contourWindow = "contours";
    cv::namedWindow(pennyWindow, cv::WindowFlags::WINDOW_NORMAL);
    cv::imshow(pennyWindow, pennyImg);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void cvObjectDetection::watershedCustomSeed()
{
    std::string roadImgName = "road_image.jpg";
    cv::Mat road, roadCopy, markerImg, segments;

    road = cv::imread(RESOURCE_PATH + roadImgName, cv::IMREAD_COLOR);
    road.copyTo(roadCopy);

    markerImg = cv::Mat::zeros(road.size(), CV_32S);
    segments = cv::Mat::zeros(road.size(), CV_8U);

}


