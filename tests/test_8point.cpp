/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues> 
#include <Eigen/SVD>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

// Based on
// "In defence of the eight point algorithm"


Eigen::Matrix3d estimateFundamentalMatrixEightPoint(Eigen::Ref<const Eigen::MatrixXd> a, Eigen::Ref<const Eigen::MatrixXd> b)
{
    // Setup system of equations Ax = 0. There will be one row in A for each correspondence.
    eigen_assert(a.cols() == b.cols());
    eigen_assert(a.rows() == b.rows());

    Eigen::Matrix<double, Eigen::Dynamic, 9> A(a.cols(), 9);

    for (Eigen::DenseIndex i = 0; i < a.cols(); ++i) {
        const auto &ca = a.col(i);
        const auto &cb = b.col(i);
        
        auto &r = A.row(i);

        r(0) = ca.x() * cb.x();     // F11
        r(1) = ca.x() * cb.y();     // F21
        r(2) = ca.x();              // F31
        r(3) = ca.y() * cb.x();     // F12
        r(4) = ca.y() * cb.y();     // F22
        r(5) = ca.y();              // F32
        r(6) = cb.x();              // F13
        r(7) = cb.y();              // F23
        r(8) = 1.0;                 // F33
    }

    // Seek for a least squares solution such that |Ax| = 1. Given by the unit eigenvector of A'A associated with the smallest eigenvalue.
    Eigen::SelfAdjointEigenSolver< Eigen::Matrix<double, Eigen::Dynamic, 9> > e;
    e.compute((A.transpose() * A));
    eigen_assert(e.info() == Eigen::Success);

    Eigen::Matrix<double, 1, 9> f = e.eigenvectors().col(0); // Sorted ascending by eigenvalue.

    Eigen::Matrix3d F;
    F <<
        f(0), f(3), f(6),
        f(1), f(4), f(7),
        f(2), f(5), f(8);

    // Enforce singularity constraint such that rank(F) = 2. Which is the closest singular matrix to F under Frobenius norm.
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::DiagonalMatrix<double, 3> dPrime(svd.singularValues()(0), svd.singularValues()(1), 0.0);
    Eigen::Matrix3d FPrime = svd.matrixU() * dPrime * svd.matrixV().transpose();

    return FPrime;   
}

Eigen::Transform<double, 2, Eigen::Affine> computeNormalizingTransform(Eigen::Ref<const Eigen::MatrixXd> a)
{
    Eigen::Vector2d mean = a.rowwise().mean();
    Eigen::Vector2d stddev = (a.colwise() - mean).array().square().rowwise().mean().sqrt();    

    Eigen::Transform<double, 2, Eigen::AffineCompact> t;
    t = Eigen::Scaling(1.0 / stddev.norm()) *  Eigen::Translation2d(-mean);
    return t;
}

Eigen::Transform<double, 3, Eigen::Affine> recoverPose(Eigen::Ref<const Eigen::MatrixXd> a, Eigen::Ref<const Eigen::MatrixXd> b, const Eigen::Matrix3d &F, const Eigen::Matrix3d &K)
{
    std::vector<cv::Point2d> p0(a.cols());
    std::vector<cv::Point2d> p1(b.cols());

    std::cout << __LINE__ << std::endl;

    for (Eigen::DenseIndex i = 0; i < a.cols(); ++i) {
        p0[i] = cv::Point2d(a(0, i), a(1, i));
        p1[i] = cv::Point2d(b(0, i), b(1, i));
    }

    std::cout << "F-OpenCV" << std::endl;
    std::cout << cv::findFundamentalMat(p0, p1) << std::endl;
   
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
    const int nPoints = 60;

    Eigen::Matrix3d k;
    k << 
        foc, 0.0, 0.5 *(width - 1),
        0.0, foc, 0.5 *(height - 1),
        0.0, 0.0, 1.0;

    // Generate random 3D points
    Eigen::AlignedBox3d box(Eigen::Vector3d(-500, -500, 300), Eigen::Vector3d(500, 500, 1500));
    Eigen::MatrixXd points(3, nPoints);
    for (int i = 0; i < nPoints; ++i) {
        points.col(i) = box.sample();
    }

    // Assume the first camera at origin and the second freely transformed.
    Eigen::Transform<double, 3, Eigen::AffineCompact> t0;
    t0.setIdentity();

    Eigen::Transform<double, 3, Eigen::AffineCompact> t1;
    t1 = Eigen::Translation3d(150.0, 0, 0);

    // Generate projected image points
    Eigen::Matrix<double, 3, 4> p0 = k * t0.matrix();
    Eigen::Matrix<double, 3, 4> p1 = k * t1.matrix();

    Eigen::Matrix<double, 2, Eigen::Dynamic> image0 = (p0 * points.colwise().homogeneous()).colwise().hnormalized();
    Eigen::Matrix<double, 2, Eigen::Dynamic> image1 = (p1 * points.colwise().homogeneous()).colwise().hnormalized();
    
    // Normalize points
    Eigen::Transform<double, 2, Eigen::Affine> tin0 = computeNormalizingTransform(image0);
    Eigen::Transform<double, 2, Eigen::Affine> tin1 = computeNormalizingTransform(image1);

    Eigen::Matrix<double, 2, Eigen::Dynamic> nimage0 = (tin0.matrix() * image0.colwise().homogeneous()).colwise().hnormalized();
    Eigen::Matrix<double, 2, Eigen::Dynamic> nimage1 = (tin1.matrix() * image1.colwise().homogeneous()).colwise().hnormalized();
    
    Eigen::Matrix3d Fn = estimateFundamentalMatrixEightPoint(nimage0, nimage1);
    Eigen::Matrix3d F = (tin1.matrix().transpose() * Fn * tin0.matrix());

    std::cout << "8Point" << std::endl;
    std::cout << F << std::endl;
    

    for (Eigen::DenseIndex i = 0; i < image0.cols(); ++i) {
        std::cout << image1.col(i).transpose().homogeneous() * F * image0.col(i).homogeneous() << std::endl;
    }    

    // Recover pose using OpenCV
    recoverPose(image0, image1, F, k);
}