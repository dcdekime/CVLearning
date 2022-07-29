#include "cvImageProcessing.hpp"

void cvImageProcessing::blendImages(cv::Mat& img1, cv::Mat& img2)
{
    // *** Convert images ***
    // cv::cvtColor(img1, img1, cv::COLOR_BGR2RGB);
    // cv::cvtColor(img2, img2, cv::COLOR_BGR2RGB);
    cv::cvtColor(img2, img2, cv::COLOR_BGR2GRAY);

    std::cout << "img1 size: " << img1.size() << std::endl;
    std::cout << "img2 size: " << img2.size() << std::endl;

    // *** Resizing images ***
    // cv::resize(img1, img1, cv::Size(500,500));
    //cv::resize(img2, img2, cv::Size(300,300));

    // std::cout << "img1 size: " << img1.size() << std::endl;
    // std::cout << "img2 size: " << img2.size() << std::endl;

    // *** Blending two images together ***
    // cv::Mat blendImg;
    // cv::addWeighted(img1, 0.8, img2, 0.1, 0, blendImg);

    // *** Paste one image on region of another image ***
    //img2.copyTo(img1(cv::Rect(50,100,img2.cols, img2.rows)));

    // *** Crop an image ***
    cv::Mat cropImg2 = img2(cv::Range(img2.rows/2, img2.rows), cv::Range(0, img2.cols/2));

    std::string windowName1 = "pup1";
    std::string windowName2 = "pup2";
    //std::string windowName3 = "blended";
    cv::namedWindow(windowName1);
    cv::namedWindow(windowName2);
    //cv::namedWindow(windowName3);

    cv::imshow(windowName1, img1);
    cv::imshow(windowName2, cropImg2);
    //cv::imshow(windowName3, blendImg);

    cv::waitKey(0);
}

void cvImageProcessing::imageThresholding()
{
    cv::Mat rainbowImg = cv::imread("/home/parallels/Documents/Vision_Stuff/resources/rainbow.jpg", 0);

    if (rainbowImg.empty())
    {
        std::cout << "cannnot find image!" << std::endl;
    }

    cv::threshold(rainbowImg, rainbowImg, 127, 255, cv::THRESH_BINARY);

    cv::imshow("rainbow", rainbowImg);
    cv::waitKey(0);
}

void cvImageProcessing::blurrSmooth()
{
    cv::Mat bricksGamma;
    cv::Mat blurredImg;
    cv::Mat brickImg = cv::imread("/home/parallels/Documents/Vision_Stuff/resources/bricks.jpg", cv::IMREAD_COLOR);
    brickImg.convertTo(brickImg, CV_32F, 1.0 / 255, 0);

    // *** alter gamma value to change brightness of an image ***
    double gamma = 0.25;
    cv::pow(brickImg, gamma, bricksGamma);

    // *** add text to image ***
    cv::HersheyFonts font = cv::FONT_HERSHEY_COMPLEX;
    cv::putText(brickImg, "bricks", cv::Point(10,600), font, 10, cv::Scalar(0, 0, 255), 4);

    // *** create kernel for blurring ***
    // cv::Mat kernel = cv::Mat::ones(cv::Size(5,5), CV_32F) / 25;
    // cv::filter2D(brickImg, blurredImg, -1, kernel);
    //cv::blur(brickImg, blurredImg, cv::Size(10,10));
    //cv::GaussianBlur(brickImg, blurredImg, cv::Size(5,5), 10);
    //cv::medianBlur(brickImg, blurredImg, 5);
    cv::bilateralFilter(brickImg, blurredImg, 9, 75, 75);

    cv::namedWindow("bricks_original");
    cv::namedWindow("bricks_blurred");
    cv::imshow("bricks_original", brickImg);
    cv::imshow("bricks_blurred", blurredImg);
    cv::waitKey(0);
}

void cvImageProcessing::morphological()
{
    // create images
    cv::Mat blankImg = cv::Mat::zeros(cv::Size(600,600), CV_64F);
    cv::Mat erodeImg;
    cv::Mat clearedImg;
    cv::Mat noiseImg(cv::Size(600,600), CV_64F);
    cv::randn(noiseImg, 0, 2);
    noiseImg = noiseImg * 255.0F;

    // create font and draw text on image
    cv::HersheyFonts font = cv::FONT_HERSHEY_SIMPLEX;
    cv::putText(blankImg, "ABCDE", cv::Point(50,300), font, 5, cv::Scalar(255,255,255));

    // combine images
    noiseImg = blankImg + noiseImg;

    // *** create morphological kernels ***
    cv::Mat erosionKernel = cv::Mat::ones(cv::Size(5,5), cv::IMREAD_COLOR);

    // *** perform morphological actions
    //cv::erode(blankImg, erodeImg, erosionKernel, cv::Point(-1, -1), 1);
    cv::morphologyEx(noiseImg, clearedImg, cv::MORPH_OPEN, erosionKernel);

    cv::namedWindow("original_img");
    //cv::namedWindow("eroded_img");
    cv::namedWindow("noise_img");
    cv::namedWindow("cleared_img");

    cv::imshow("original_img", blankImg);
    //cv::imshow("eroded_img", erodeImg);
    cv::imshow("noise_img", noiseImg);
    cv::imshow("cleared_img", clearedImg);
    cv::waitKey(0);
}

void cvImageProcessing::gradients()
{
    cv::Mat sudokuImg = cv::imread("/home/parallels/Documents/Vision_Stuff/resources/sudoku.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat sobelX;
    cv::Mat sobelY;
    cv::Mat laplace;
    cv::Mat blended;
    cv::Mat blendedGradient;

    // *** perform sobel operations -> image gradients x and y ***
    cv::Sobel(sudokuImg, sobelX, CV_64F, 1, 0, 5);
    cv::Sobel(sudokuImg, sobelY, CV_64F, 0, 1, 5);
    cv::Laplacian(sudokuImg, laplace, CV_64F);
    cv::addWeighted(sobelX, 0.5, sobelY, 0.5, 0.0, blended);

    cv::Mat gradientKernel = cv::getStructuringElement(0, cv::Size(4,4));
    cv::morphologyEx(blended, blendedGradient, cv::MORPH_GRADIENT, gradientKernel);

    cv::namedWindow("original");
    // cv::namedWindow("Gx");
    // cv::namedWindow("Gy");
    // cv::namedWindow("laplace");
    //cv::namedWindow("blendedGXGY");
    cv::namedWindow("blendedGradient");

    cv::imshow("original", sudokuImg);
    // cv::imshow("Gx", sobelX);
    // cv::imshow("Gy", sobelY);
    // cv::imshow("laplace", laplace);
    //cv::imshow("blendedGXGY", blended);
    cv::imshow("blendedGradient", blendedGradient);

    cv::waitKey(0);
    cv::destroyAllWindows();
}

void drawHistogram(cv::Mat& hist_blue, cv::Mat& hist_green, cv::Mat& hist_red)
{
    const int histSize = 256;
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::normalize(hist_blue, hist_blue, 0, histImage.rows, cv::NORM_MINMAX, -1,
                  cv::Mat());
    cv::normalize(hist_green, hist_green, 0, histImage.rows, cv::NORM_MINMAX, -1,
                  cv::Mat());
    cv::normalize(hist_red, hist_red, 0, histImage.rows, cv::NORM_MINMAX, -1,
                  cv::Mat());

    for (int i = 1; i < histSize; i++) {
      cv::line(
          histImage,
          cv::Point(bin_w * (i - 1), hist_h - cvRound(hist_blue.at<float>(i - 1))),
          cv::Point(bin_w * (i), hist_h - cvRound(hist_blue.at<float>(i))),
          cv::Scalar(255, 0, 0), 2, 8, 0);
      cv::line(
          histImage,
          cv::Point(bin_w * (i - 1), hist_h - cvRound(hist_green.at<float>(i - 1))),
          cv::Point(bin_w * (i), hist_h - cvRound(hist_green.at<float>(i))),
          cv::Scalar(0, 255, 0), 2, 8, 0);
      cv::line(
          histImage,
          cv::Point(bin_w * (i - 1), hist_h - cvRound(hist_red.at<float>(i - 1))),
          cv::Point(bin_w * (i), hist_h - cvRound(hist_red.at<float>(i))),
          cv::Scalar(0, 0, 255), 2, 8, 0);
    }

    cv::namedWindow("calcHist Demo", cv::WindowFlags::WINDOW_NORMAL);
    cv::imshow("calcHist Demo", histImage);
    //cv::waitKey(0);
}

void cvImageProcessing::histogram()
{
    cv::Mat dark_horse = cv::imread("/home/parallels/Documents/Vision_Stuff/resources/horse.jpg", cv::IMREAD_COLOR);
    cv::Mat rainbow = cv::imread("/home/parallels/Documents/Vision_Stuff/resources/rainbow.jpg", cv::IMREAD_COLOR);
    cv::Mat blueBricks = cv::imread("/home/parallels/Documents/Vision_Stuff/resources/bricks.jpg", cv::IMREAD_COLOR);
    cv::Mat histogramMat_blue, histogramMat_green, histogramMat_red;
    std::vector<cv::Mat> srcImages = {dark_horse, rainbow, blueBricks};

    int imgCount = 1;
    for (auto& srcImg : srcImages)
    {
        std::string windowName = "src " + std::to_string(imgCount);
        cv::namedWindow(windowName, cv::WindowFlags::WINDOW_NORMAL);
        cv::imshow(windowName, srcImg);

        // *** split source image into BGR planes ***
        std::vector<cv::Mat> bgrPlanes;
        cv::split(srcImg, bgrPlanes);

        // *** calculate histogram ***
        const int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};
        bool uniform = true;
        bool accumulate = false;

        cv::calcHist(&bgrPlanes[0], 1, 0, cv::Mat(), histogramMat_blue, 1, &histSize,
                    &histRange, uniform, accumulate);

        cv::calcHist(&bgrPlanes[1], 1, 0, cv::Mat(), histogramMat_green, 1, &histSize,
                    &histRange, uniform, accumulate);

        cv::calcHist(&bgrPlanes[2], 1, 0, cv::Mat(), histogramMat_red, 1, &histSize,
                    &histRange, uniform, accumulate);
        
        drawHistogram(histogramMat_blue, histogramMat_green, histogramMat_red);

        if (cv::waitKey(0) == 27)
        {
            cv::destroyAllWindows();
            break;
        }

        imgCount++;
    }

    // *** Histogram Equalization ***
    
}