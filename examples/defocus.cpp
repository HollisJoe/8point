/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#include <eight/fundamental.h>
#include <eight/normalize.h>
#include <eight/distance.h>
#include <eight/essential.h>
#include <eight/project.h>
#include <eight/select.h>
#include <random>
#include <iostream>

#include <opencv2/opencv.hpp>

class KLT {
public:

    KLT(const cv::Mat &img, const std::vector<cv::Point2f> &initial)
        :_next(initial)
    {
        cv::cvtColor(img, _nextGray, CV_BGR2GRAY);
        _nextStatus.resize(initial.size(), 255);
    }

    void update(cv::Mat &img) {        
        
        std::swap(_prev, _next);
        std::swap(_prevGray, _nextGray);
        std::swap(_prevStatus, _nextStatus);

        cv::cvtColor(img, _nextGray, CV_BGR2GRAY);
        cv::calcOpticalFlowPyrLK(_prevGray, _nextGray, _prev, _next, _nextStatus, _err);

        for (size_t i = 0; i < _nextStatus.size(); ++i) {
            _nextStatus[i] &= _prevStatus[i];
        }        
    }

    std::vector<cv::Point2f> &location() {
        return _next;
    }

    std::vector<uchar> &status() {
        return _nextStatus;
    }


private:
    std::vector<cv::Point2f> _prev, _next;
    cv::Mat _prevGray, _nextGray;
    std::vector<uchar> _prevStatus, _nextStatus;
    std::vector<float> _err;
};

Eigen::MatrixXd toEight(const std::vector<cv::Point2f> &x) {
    Eigen::MatrixXd m(2, x.size());

    for (size_t i = 0; i < x.size(); ++i) {
        m(0, i) = x[i].x;
        m(1, i) = x[i].y;
    }
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

    // Assume first frame is reference frame
    cv::Mat ref, refGray;
    vc >> ref;
    cv::cvtColor(ref, refGray, CV_BGR2GRAY);

    std::vector<cv::Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;
    int maxCorners = 500;

    cv::goodFeaturesToTrack(refGray,
        corners,
        maxCorners,
        qualityLevel,
        minDistance,
        cv::Mat(),
        blockSize,
        useHarrisDetector,
        k);

    KLT klt(ref, corners);

    std::vector< std::vector<cv::Point2f> > trackedLocations;
    std::vector< std::vector<uchar> > trackedStatus;

    trackedLocations.push_back(klt.location());
    trackedStatus.push_back(klt.status());

    cv::Mat f;
    while (vc.grab()) {
        vc.retrieve(f);
        klt.update(f);

        trackedLocations.push_back(klt.location());
        trackedStatus.push_back(klt.status());

        std::vector<cv::Point2f> &loc = klt.location();
        std::vector<uchar> &status = klt.status();

        for (int i = 0; i < loc.size(); i++)
        {
            if (status[i]) {
                cv::circle(f, loc[i], 2, cv::Scalar(0, 255, 0), -1, 8, 0);
            }
            else {
                cv::circle(f, loc[i], 2, cv::Scalar(0, 0, 255), -1, 8, 0);
            }
        }

        cv::imshow("f", f);
        cv::waitKey(60);

    }
         

    
}
