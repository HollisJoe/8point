/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#include "catch.hpp"

#include <eight/project.h>
#include <eight/triangulate.h>
#include "utils.h"

#include <opencv2\opencv.hpp>

TEST_CASE("test_triangulation")
{
    const double foc = 530.0;
    const int width = 640;
    const int height = 480;
    const int nPoints = 60;

    Eigen::Matrix3d k;
    k << 
        foc, 0.0, 0.5 *(width - 1),
        0.0, foc, 0.5 *(height - 1),
        0.0, 0.0, 1.0;

    // Generate random 3D points
    Eigen::Matrix<double, 3, Eigen::Dynamic> points = eight::utils::samplePointsInBox(Eigen::Vector3d(-500, -500, 300), Eigen::Vector3d(500, 500, 1500), nPoints);

    // Assume the first camera at origin and the second freely transformed.
    Eigen::AffineCompact3d t0;
    t0.setIdentity();

    Eigen::AffineCompact3d t1;
    t1 = Eigen::Translation3d(15.0, 0.0, 3.5) * Eigen::AngleAxisd(0.25*M_PI, Eigen::Vector3d(0.5, -0.3, 0.2).normalized());

    // Generate projected image points
    Eigen::Matrix<double, 3, 4> cam0 = eight::cameraMatrix(k, t0);
    Eigen::Matrix<double, 3, 4> cam1 = eight::cameraMatrix(k, t1);

    Eigen::Matrix<double, 2, Eigen::Dynamic> image0 = eight::perspectiveProject(points, cam0).colwise().hnormalized();
    Eigen::Matrix<double, 2, Eigen::Dynamic> image1 = eight::perspectiveProject(points, cam1).colwise().hnormalized();

    Eigen::Matrix<double, 3, Eigen::Dynamic > pointsTriangulated = eight::triangulateMany(cam0, cam1, image0, image1);

    REQUIRE(pointsTriangulated.isApprox(points));

}