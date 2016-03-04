/**
 This file is part of 8point.
 
 Copyright(C) 2015/2016 Christoph Heindl
 All rights reserved.
 
 This software may be modified and distributed under the terms
 of the BSD license.See the LICENSE file for details.
*/

#include <eight/fundamental.h>
#include <eight/normalize.h>

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

namespace eight {
    
    Eigen::Matrix3d findFundamentalMatrix(Eigen::Ref<const Eigen::MatrixXd> a, Eigen::Ref<const Eigen::MatrixXd> b) {
        
        eigen_assert(a.cols() == b.cols());
        eigen_assert(a.rows() == b.rows());
        eigen_assert(a.cols() >= 8);
        
        // Setup system of equations Ax = 0. There will be one row in A for each correspondence.
        Eigen::Matrix<double, Eigen::Dynamic, 9> A(a.cols(), 9);
        
        for (Eigen::DenseIndex i = 0; i < a.cols(); ++i) {
            const auto &ca = a.col(i);
            const auto &cb = b.col(i);
            
            auto r = A.row(i);
            
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
    
    Eigen::Matrix3d findFundamentalMatrixNormalized(Eigen::Ref<const Eigen::MatrixXd> a, Eigen::Ref<const Eigen::MatrixXd> b)
    {
        Eigen::Transform<double, 2, Eigen::Affine> t0 = findIsotropicNormalizingTransform(a);
        Eigen::Transform<double, 2, Eigen::Affine> t1 = findIsotropicNormalizingTransform(b);
        
        Eigen::Matrix<double, 2, Eigen::Dynamic> na = (t0.matrix() * a.colwise().homogeneous()).colwise().hnormalized();
        Eigen::Matrix<double, 2, Eigen::Dynamic> nb = (t1.matrix() * b.colwise().homogeneous()).colwise().hnormalized();
        
        Eigen::Matrix3d Fn = eight::findFundamentalMatrix(na, nb);
        Eigen::Matrix3d F = (t1.matrix().transpose() * Fn * t0.matrix());
        return F;
    }
}