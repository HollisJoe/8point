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
#include <random>

#ifdef _WIN32
#define GOOGLE_GLOG_DLL_DECL
#pragma warning(push)
#pragma warning(disable : 4251 4355)
#include <ceres/ceres.h>
#include "ceres/rotation.h"
#include <glog/logging.h>
#pragma warning(pop)
#else
#include <ceres/ceres.h>
#include "ceres/rotation.h"
#include <glog/logging.h>
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

Eigen::MatrixXd imagePointsToRetinaPoints(const std::vector<cv::Point2f> &p, const Eigen::Matrix3d &kinv)
{
    Eigen::MatrixXd r(3, p.size());

    for (size_t i = 0; i < p.size(); ++i) {
        Eigen::Vector3d x(p[i].x, p[i].y, 1.0);
        r.col(i) = kinv * x;
    }

    return r;
}

void findFeaturesInReference(cv::Mat &gray, std::vector<cv::Point2f> &corners) {
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 5;
    bool useHarrisDetector = false;
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

struct GeneralReprojectionError {
    GeneralReprojectionError(const double *no, const double *np)
        : _no(no), _np(np) 
    {}

    template <typename T>
    bool operator()(
        const T* const camera,
        const T* const inverseDepth,
        T* residuals) const 
    {

        T p[3] = { _np[0] / inverseDepth[0], _np[1] / inverseDepth[0], _np[2] / inverseDepth[0] };

        const T a[2] = {
            p[0] - camera[2] * p[1] + camera[1],
            p[1] - camera[0] + camera[2] * p[0]
        };

        const T c = -camera[1] * p[0] + camera[0] * p[1] + T(1);

        const T e[2] = {
            _no[0] * c - a[0],
            _no[1] * c - a[1]
        };


        const T f[2] = {
            _no[0] * camera[5] - camera[3],
            _no[1] * camera[5] - camera[4]
        };

        residuals[0] = (e[0] + f[0] * inverseDepth[0]) / (c + camera[5] * inverseDepth[0]);
        residuals[1] = (e[1] + f[1] * inverseDepth[0]) / (c + camera[5] * inverseDepth[0]);

        return true;
    }

    static ceres::CostFunction* Create(const double *observed, const double *point)
    {
        return (new ceres::AutoDiffCostFunction<GeneralReprojectionError, 2, 6, 1>(
            new GeneralReprojectionError(observed, point)));
    }

    const double *_no;
    const double *_np;
};

int main(int argc, char **argv) {

    google::InitGoogleLogging(argv[0]);

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

    Eigen::Matrix3d invk = k.inverse();


    // Detect trackable features in reference frame
    cv::Mat ref, refGray;
    vc >> ref;
    cv::cvtColor(ref, refGray, CV_BGR2GRAY);

    std::cout << __LINE__ << std::endl;
    std::vector< Eigen::MatrixXd > retinapoints;
    std::vector<cv::Point2f> corners;
    
    std::cout << __LINE__ << std::endl;
    findFeaturesInReference(refGray, corners);
    std::vector<uchar> status(corners.size(), 1);
    
    std::cout << __LINE__ << std::endl;
    retinapoints.push_back(imagePointsToRetinaPoints(corners, invk));

    std::cout << __LINE__ << std::endl;
    
    

    int frameCount = 0;

    cv::Mat f;
    while (vc.grab()) {
        vc.retrieve(f);

        std::vector<cv::Point2f> loc;
        std::vector<uchar> s;

        trackFeatures(ref, corners, f, loc, s);

        retinapoints.push_back(imagePointsToRetinaPoints(loc, invk));
        for (size_t i = 0; i < s.size(); ++i) {
            status[i] &= s[i];

            if (status[i]) {
                cv::circle(f, loc[i], 2, cv::Scalar(0, 255, 0));
            }
        }
        
        cv::imshow("track", f);
        cv::waitKey(10);
    }
    
    std::cout << __LINE__ << std::endl;

    std::uniform_real_distribution<double> dist(10.0, 20.0);
    std::default_random_engine re;
    std::vector<double> idepths(status.size(), 10.0);

    /*
    for (size_t i = 0; i < status.size(); ++i) {
        idepths.push_back(1.0 / dist(re));
    }*/

    std::copy(idepths.begin(), idepths.end(), std::ostream_iterator<double>(std::cout, ","));

    // Setup camera parameters
    Eigen::MatrixXd camparams(6, retinapoints.size());
    camparams.setZero();

    // Setup nnls system
    ceres::Problem problem;

    // For each camera
    for (size_t f = 1; f < retinapoints.size(); ++f) {
        // For each point observed
        for (Eigen::DenseIndex p = 0; p < retinapoints[f].cols(); ++p) {
            if (!status[p])
                continue;

            ceres::CostFunction* cost_function = GeneralReprojectionError::Create(
                retinapoints[f].col(p).data(),
                retinapoints[0].col(p).data()
            );

            problem.AddResidualBlock(
                cost_function,
                NULL /* squared loss */,
                camparams.col(f).data(),
                &idepths[p]);
        }

        

    }

    std::cout << __LINE__ << std::endl;


    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 400;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    std::ofstream ofs("points.xyz");
    for (Eigen::DenseIndex i = 0; i < retinapoints[0].cols(); ++i) {
        if (!status[i])
            continue;

        Eigen::Vector3d x = retinapoints[0].col(i) * idepths[i];

        ofs << x(0) << " " << x(1) << " " << x(2) << std::endl;
    }
    ofs.close();

    std::cout << camparams.col(0) << std::endl;



    
}
