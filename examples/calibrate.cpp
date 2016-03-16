/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#include <iostream>
#include <opencv2/opencv.hpp>

void calcBoardCornerPositions(cv::Size boardSize, float squareSize, std::vector<cv::Point3f>& corners)
{
    corners.clear();
    for (int i = 0; i < boardSize.height; ++i)
        for (int j = 0; j < boardSize.width; ++j)
            corners.push_back(cv::Point3f(j*squareSize, i*squareSize, 0));
}

int main(int argc, char **argv) {

    if (argc != 2) {
        std::cerr << argv[0] << " videofile" << std::endl;
        return -1;
    }

    cv::VideoCapture vc;
    if (!vc.open(argv[1])) {
        std::cerr << "Failed to open video" << std::endl;
        return -1;
    }


    std::vector<cv::Mat> imgs;

    cv::Mat img, imgShow;
    while (vc.grab()) {
        vc.retrieve(img);

        cv::resize(img, imgShow, cv::Size(), 0.5, 0.5);
        cv::imshow("Image", imgShow);
        int key = cv::waitKey();

        if (key == 's') {
            imgs.push_back(img.clone());
        }
    }


    std::vector<std::vector<cv::Point2f> > imagePoints;   

    const int chessBoardFlags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
    const cv::Size boardSize(10, 7);
    const float squareSize = 33.85f;

    for (size_t i = 0; i < imgs.size(); ++i) {
        std::vector<cv::Point2f> points;
        bool found = cv::findChessboardCorners(imgs[i], boardSize, points, chessBoardFlags);

        if (found) {
            cv::Mat viewGray;
            cv::cvtColor(imgs[i], viewGray, cv::COLOR_BGR2GRAY);
            cornerSubPix(viewGray, points, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
            imagePoints.push_back(points);
        }
        cv::drawChessboardCorners(imgs[i], boardSize, cv::Mat(points), found);
        cv::imshow("chessboard", imgs[i]);
        cv::waitKey(100);
    }

    // Perform calibration
    std::vector<std::vector<cv::Point3f> > objectPoints(1);
    calcBoardCornerPositions(boardSize, squareSize, objectPoints[0]);
    objectPoints.resize(imagePoints.size(), objectPoints[0]);

    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64FC1);
    cv::Mat distCoeffs = cv::Mat::zeros(8, 1, CV_64FC1);
    double rms = cv::calibrateCamera(objectPoints, imagePoints, imgs[0].size(), cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_FIX_K3 | CV_CALIB_FIX_K2 | CV_CALIB_FIX_ASPECT_RATIO);

    std::cout << "RMS " << rms << std::endl;
    std::cout << cameraMatrix << std::endl;
    std::cout << distCoeffs << std::endl;
    
}
