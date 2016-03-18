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
#include <eight/triangulate.h>
#include <random>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <fstream>

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4251 4355)
#include <ceres/ceres.h>
#pragma warning(pop)
#else
#include <ceres/ceres.h>
#endif



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

    return m;
}

void findFeaturesInReference(cv::Mat &gray, std::vector<cv::Point2f> &corners) {
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 5;
    bool useHarrisDetector = true;
    double k = 0.04;
    int maxCorners = 500;

    cv::goodFeaturesToTrack(gray,
        corners,
        maxCorners,
        qualityLevel,
        minDistance,
        cv::Mat(),
        blockSize,
        useHarrisDetector,
        k);
}

void trackFeatures(cv::Mat ref, const std::vector<cv::Point2f> &refLocs, cv::Mat target, std::vector<cv::Point2f> &targetLocs, std::vector<uchar> &status)
{
    KLT klt(ref, refLocs);
    klt.update(target);
    targetLocs = klt.location();
    status = klt.status();
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

    // Taken from http://yf.io/p/tiny/
    // Use 8point_defocus.exe "stream\stone6_still_%04d.png"
    
    Eigen::Matrix3d k;
    k <<
        1781.0, 0.0, 960.0,
        0.0, 1781.0, 540.0,
        0.0, 0.0, 1.0;


    // Assume first frame is reference frame
    cv::Mat ref, refGray;
    vc >> ref;
    cv::cvtColor(ref, refGray, CV_BGR2GRAY);

    std::vector<cv::Point2f> corners;
    findFeaturesInReference(refGray, corners);

    std::vector< std::vector<cv::Point2f> > trackedLocations;
    std::vector<uchar> status(corners.size(), true);

    int frameCount = 0;

    cv::Mat f;
    while (vc.grab()) {
        vc.retrieve(f);
       
        std::vector<cv::Point2f> loc;
        std::vector<uchar> s;

        trackFeatures(ref, corners, f, loc, s);
        for (size_t i = 0; i < s.size(); ++i) {
            status[i] &= s[i];
        }
        trackedLocations.push_back(loc);

        ++frameCount;

        for (int i = 0; i < loc.size(); i++)
        {
            if (status[i]) {
                cv::circle(f, loc[i], 2, cv::Scalar(0, 255, 0), -1, 8, 0);
            }
            else {
                cv::circle(f, loc[i], 2, cv::Scalar(0, 0, 255), -1, 8, 0);
            }
        }

        Eigen::Matrix<double, 2, Eigen::Dynamic> image0 = toEight(trackedLocations[0]);
        Eigen::Matrix<double, 2, Eigen::Dynamic> image1 = toEight(loc);

        std::vector<Eigen::DenseIndex> inliers;
        Eigen::Matrix3d F = eight::fundamentalMatrixRobust(image0, image1, inliers, 1.0);
        Eigen::Matrix3d E = eight::essentialMatrix(k, F);

        image0 = eight::selectColumnsByIndex(image0, inliers.begin(), inliers.end());
        image1 = eight::selectColumnsByIndex(image1, inliers.begin(), inliers.end());

        Eigen::Matrix<double, 3, 4> pose = eight::pose(E, k, image0, image1);
        
        Eigen::Matrix<double, 3, 4> cam0 = eight::cameraMatrix(k, Eigen::Matrix<double, 3, 4>::Identity());
        Eigen::Matrix<double, 3, 4> cam1 = eight::cameraMatrix(k, pose);
        Eigen::Matrix<double, 3, Eigen::Dynamic > points = eight::triangulateMany(cam0, cam1, image0, image1);


        std::stringstream str;
        str << "Frame_" << frameCount << ".xyz";

        std::ofstream ofs(str.str());
        for (Eigen::DenseIndex i = 0; i < points.cols(); ++i) {
            ofs << points(0, i) << " " << points(1, i) << " " << points(2, i) << std::endl;
        }

        ofs.close();


        cv::imshow("f", f);
        cv::waitKey(60);

    }
         

    
}
