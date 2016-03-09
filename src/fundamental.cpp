/**
 This file is part of 8point.
 
 Copyright(C) 2015/2016 Christoph Heindl
 All rights reserved.
 
 This software may be modified and distributed under the terms
 of the BSD license.See the LICENSE file for details.
*/

#include <eight/fundamental.h>
#include <eight/normalize.h>
#include <eight/distance.h>
#include <eight/select.h>

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <vector>
#include <random>

namespace eight {
    
    Eigen::Matrix3d fundamentalMatrixUnnormalized(Eigen::Ref<const Eigen::MatrixXd> a, Eigen::Ref<const Eigen::MatrixXd> b) {
        
        eigen_assert(a.cols() == b.cols());
        eigen_assert(a.rows() == b.rows());
        eigen_assert(a.cols() >= 8);
        
        // Setup system of equations Ax = 0. There will be one row in A for each correspondence.
        Eigen::Matrix<double, Eigen::Dynamic, 9> A(a.cols(), 9);
        
        for (Eigen::DenseIndex i = 0; i < a.cols(); ++i) {
            const auto &ca = a.col(i);
            const auto &cb = b.col(i);
            
            auto r = A.row(i);
            
            r(0) = cb.x() * ca.x();     // F11
            r(1) = cb.x() * ca.y();     // F21
            r(2) = cb.x();              // F31
            r(3) = cb.y() * ca.x();     // F12
            r(4) = cb.y() * ca.y();     // F22
            r(5) = cb.y();              // F32
            r(6) = ca.x();              // F13
            r(7) = ca.y();              // F23
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
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::DiagonalMatrix<double, 3> dPrime(svd.singularValues()(0), svd.singularValues()(1), 0.0);
        Eigen::Matrix3d FPrime = svd.matrixU() * dPrime * svd.matrixV().transpose();
        
        return FPrime;   

    }
    
    Eigen::Matrix3d fundamentalMatrix(Eigen::Ref<const Eigen::MatrixXd> a, Eigen::Ref<const Eigen::MatrixXd> b)
    {
        Eigen::Transform<double, 2, Eigen::Affine> t0 = findIsotropicNormalizingTransform(a);
        Eigen::Transform<double, 2, Eigen::Affine> t1 = findIsotropicNormalizingTransform(b);
        
        Eigen::Matrix<double, 2, Eigen::Dynamic> na = (t0.matrix() * a.colwise().homogeneous()).colwise().hnormalized();
        Eigen::Matrix<double, 2, Eigen::Dynamic> nb = (t1.matrix() * b.colwise().homogeneous()).colwise().hnormalized();
        
        Eigen::Matrix3d Fn = eight::fundamentalMatrixUnnormalized(na, nb);
        Eigen::Matrix3d F = (t0.matrix().transpose() * Fn * t1.matrix());
        return F;
    }

    template <class ForwardIterator, class T>
    void fillIncremental(ForwardIterator first, ForwardIterator last, T val)
    {
        while (first != last) {
            *first = val;
            ++first;
            ++val;
        }
    }

    std::vector<Eigen::DenseIndex> samplePointIndices(Eigen::DenseIndex setSize, Eigen::DenseIndex sampleSize) {
        std::vector<Eigen::DenseIndex> ids(setSize);
        fillIncremental(ids.begin(), ids.end(), 0);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // Fisher-Yates sampling
        std::vector<Eigen::DenseIndex> result(sampleSize);
        for (Eigen::DenseIndex i = 0; i < sampleSize; i++) {
            std::uniform_int_distribution<Eigen::DenseIndex> dis(i, setSize - 1);
            std::swap(ids[i], ids[dis(gen)]);
            result[i] = ids[i];
        }
        
        return result;
    }

    Eigen::Matrix3d fundamentalMatrixRobust(Eigen::Ref<const Eigen::MatrixXd> a, Eigen::Ref<const Eigen::MatrixXd> b, std::vector<Eigen::DenseIndex> &inliers, double d, double e, double p)
    {
        // First perform a ransac to find a a fundamental matrix. Then re-estimate fundamental matrix from all inliers.

        int niter = static_cast<int>(std::ceil(std::log(1.0 - p) / std::log(1.0 - std::pow(1.0 - e, 8))));


        const double d2 = d * d;

        Eigen::Matrix3d bestF = Eigen::Matrix3d::Constant(std::numeric_limits<double>::quiet_NaN());
        inliers.clear();

        for (int iter = 0; iter < niter; ++iter) {
            // Randomly select 8 different points
            std::vector<Eigen::DenseIndex> modelIds = samplePointIndices(a.cols(), 8);
            
            // Build model
            Eigen::Matrix<double, 2, Eigen::Dynamic> sa = selectColumnsByIndex(a, modelIds.begin(), modelIds.end());
            Eigen::Matrix<double, 2, Eigen::Dynamic> sb = selectColumnsByIndex(b, modelIds.begin(), modelIds.end());
            Eigen::Matrix3d F = fundamentalMatrix(sa, sb);

            // Evaluate inliers
            Eigen::VectorXd dists = distances(F, a, b, SampsonDistanceSquared());            
            std::vector<Eigen::DenseIndex> in;
            for (Eigen::DenseIndex i = 0; i < dists.size(); ++i) {
                if (dists(i) <= d2) {
                    in.push_back(i);
                }
            }

            if (in.size() > inliers.size()) {
                std::swap(in, inliers);
                std::swap(F, bestF);
            }
        }

        if (bestF.allFinite()) {
            // Re-estimate using all inliers
            Eigen::Matrix<double, 2, Eigen::Dynamic> sa = selectColumnsByIndex(a, inliers.begin(), inliers.end());
            Eigen::Matrix<double, 2, Eigen::Dynamic> sb = selectColumnsByIndex(b, inliers.begin(), inliers.end());
            bestF = fundamentalMatrix(sa, sb);
        }

        return bestF;
    }
}