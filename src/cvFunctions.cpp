#include "cvFunctions.hpp"

cvFunctions::cvFunctions() 
    : mImg(cv::Mat::zeros(400,400,CV_8UC3)) {};

cvFunctions::cvFunctions(cv::Mat& img) : mImg(img) {};

void cvFunctions::importImage()
{
    cv::Mat image;
    std::string phtevenImg = "phteven.jpg";
    image = cv::imread(RESOURCE_PATH + phtevenImg, cv::IMREAD_COLOR);

    if(! image.data)
    {
        std::cout<<"COULD NOT OPEN FILE" << std::endl;
        return;
    } 
       
    cv::namedWindow("Dog Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Dog Image", image);
    cv::waitKey(0);
}

void cvFunctions::captureWebcamVideo()
{
    cv::VideoCapture cap(0);

    if (cap.isOpened() == false)  
    {
        std::cout << "Cannot open the video camera" << std::endl;
        std::cin.get(); //wait for any key press
        return;
    }

    double dWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
    double dHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video
    double fps = cap.get(cv::CAP_PROP_XI_FRAMERATE); // get frames per second
    std::cout << "Resolution of the video : " << dWidth << " x " << dHeight << std::endl;
    std::cout << "FPS: " << fps << std::endl;

    bool writeVideo = false;
    std::string myVideo = "myVideo.mp4";
    cv::VideoWriter videoWriter(RESOURCE_PATH + myVideo, cv::VideoWriter::fourcc('X','V','I','D'), 25, cv::Size(dWidth, dHeight));

    while (true)
    {
        cv::Mat frame;
        bool bSuccess = cap.read(frame); // read a new frame from video 

        // Breaking the while loop if the frames cannot be captured
        if (bSuccess == false) 
        {
            std::cout << "Video camera is disconnected" << std::endl;
            std::cin.get(); //Wait for any key press
            break;
        }


        // *** perform actions ***
        int startX = int(dWidth / 2);
        int startY = int(dHeight / 2);
        int stepX = int(dWidth / 4);
        int stepY = int(dHeight / 4);

        cv::rectangle(frame, cv::Point(startX, startY), cv::Point(startX+stepX, startY+stepY), cv::Scalar(0,0,255), 4);

        // show the frame in the created window
        std::string window_name = "My Camera Feed";
        cv::namedWindow(window_name); //create a window called "My Camera Feed"
        imshow(window_name, frame);

        if (writeVideo)
            videoWriter.write(frame);

        // wait for for 10 ms until any key is pressed.  
        // If the 'Esc' key is pressed, break the while loop.
        // If the any other key is pressed, continue the loop 
        // If any key is not pressed withing 10 ms, continue the loop 
        if (cv::waitKey(10) == 27)
        {
            std::cout << "Esc key is pressed by user. Stoppig the video" << std::endl;
            break;
        }
    }

    if (writeVideo)
        videoWriter.release();

    cap.release();
    cv::destroyAllWindows();
}

// global variables
cv::Mat blankImage = cv::Mat::zeros(400,400,CV_8UC3);
bool drawing = false; // True while mnouse button down, False while mnouse buttom up
int ix = -1;
int iy = -1;

void mouseCallback(int event, int x, int y, int flags, void* userdata)
{
    if  (event == cv::EVENT_LBUTTONDOWN)
     {
        // std::cout << "DRAWING CIRCLE AT: (" << x << ", " << y << ")" << std::endl;
        // cv::circle(blankImage, cv::Point(x,y), 20,
        // cv::Scalar(0, 0, 255),
        // cv::FILLED,
        // cv::LINE_8);
        drawing = true;
        ix = x;
        iy = y;
     }
     else if (event == cv::EVENT_LBUTTONUP)
     {
        drawing = false;
        cv::rectangle(blankImage, cv::Point(ix,iy), cv::Point(x,y), cv::Scalar(0,255,0), cv::FILLED);
     }
     else if (event == cv::EVENT_RBUTTONDOWN)
     {
          std::cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
     }
     else if (event == cv::EVENT_MBUTTONDOWN)
     {
          std::cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
     }
     else if (event == cv::EVENT_MOUSEMOVE)
     {
          //std::cout << "Mouse move over the window - position (" << x << ", " << y << ")" << std::endl;
          if (drawing)
            cv::rectangle(blankImage, cv::Point(ix,iy), cv::Point(x,y), cv::Scalar(0,255,0), cv::FILLED);
     }
}

void cvFunctions::drawOnImage()
{
    std::string windowName = "myImage";
    cv::namedWindow(windowName);

    cv::setMouseCallback(windowName, mouseCallback, NULL);

    while (true)
    {
        cv::imshow(windowName, blankImage);

        if (cv::waitKey(10) == 27)
        {
            std::cout << "Esc key is pressed by user. Stoppig the video" << std::endl;
            break;
        }
    }

    cv::destroyAllWindows();
}