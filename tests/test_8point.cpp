/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#define CATCH_CONFIG_MAIN
#include "catch.hpp"


#include <eight/fundamental.h>
#include <eight/normalize.h>
#include <eight/distance.h>
#include "utils.h"


#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

// Based on
// "In defence of the eight point algorithm"


Eigen::Transform<double, 3, Eigen::Affine> recoverPose(Eigen::Ref<const Eigen::MatrixXd> a, Eigen::Ref<const Eigen::MatrixXd> b, const Eigen::Matrix3d &F, const Eigen::Matrix3d &K)
{
    std::vector<cv::Point2d> p0(a.cols());
    std::vector<cv::Point2d> p1(b.cols());

    for (Eigen::DenseIndex i = 0; i < a.cols(); ++i) {
        p0[i] = cv::Point2d(a(0, i), a(1, i));
        p1[i] = cv::Point2d(b(0, i), b(1, i));
    }
   
    double f = K(0, 0);
    cv::Point2d p(K(0, 2), K(1, 2));

    // Essential matrix from fundamental
    Eigen::Matrix3d E = K.transpose() * F * K;

    cv::Mat ECV(3,3, CV_64FC1);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            ECV.at<double>(i, j) = E(i, j);

    
    cv::Mat R, t;
    cv::recoverPose(ECV, p0, p1, R, t, f, p);

    std::cout << R << std::endl;
    std::cout << t << std::endl;
    
    return Eigen::Transform<double, 3, Eigen::Affine>();
}

TEST_CASE("8point")
{
    const double foc = 530.0;
    const int width = 640;
    const int height = 480;
    const int nPoints = 8;

    Eigen::Matrix3d k;
    k << 
        foc, 0.0, 0.5 *(width - 1),
        0.0, foc, 0.5 *(height - 1),
        0.0, 0.0, 1.0;

    // Generate random 3D points
    Eigen::Matrix<double, 3, Eigen::Dynamic> points = eight::utils::samplePointsInBox(Eigen::Vector3d(-500, -500, 300), Eigen::Vector3d(500, 500, 1500), nPoints);

    // Assume the first camera at origin and the second freely transformed.
    Eigen::Transform<double, 3, Eigen::AffineCompact> t0;
    t0.setIdentity();

    Eigen::Transform<double, 3, Eigen::AffineCompact> t1;
    t1 = Eigen::Translation3d(0.0, 10.0, 0) * Eigen::AngleAxisd(0.25*M_PI, Eigen::Vector3d::UnitX());

    // Generate projected image points
    Eigen::Matrix<double, 3, 4> p0 = k * t0.matrix();
    Eigen::Matrix<double, 3, 4> p1 = k * t1.matrix();

    Eigen::Matrix<double, 2, Eigen::Dynamic> image0 = eight::utils::projectPoints(k, t0, points);
    Eigen::Matrix<double, 2, Eigen::Dynamic> image1 = eight::utils::projectPoints(k, t1, points);
    
    // Normalize points
    Eigen::Matrix3d F = eight::findFundamentalMatrixNormalized(image0, image1);
    std::cout << "F: " << F << std::endl;
    
    Eigen::VectorXd dists = eight::distances(F, image0, image1, eight::SampsonDistance());
    std::cout << "Mean distance " << dists.mean() << std::endl;

    // Recover pose using OpenCV
    recoverPose(image0, image1, F, k);
    std::cout << t1.matrix() << std::endl;
}