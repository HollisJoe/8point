/**
    This file is part of 8point.

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#ifndef EIGHT_ERROR_H
#define EIGHT_ERROR_H

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace eight {
    
    /** 
        First order approximation of geometric error.
    */
    class SampsonDistanceSquared {
    public:
        double operator()(const Eigen::Matrix3d &f, Eigen::Ref<const Eigen::Vector2d> a,  Eigen::Ref<const Eigen::Vector2d> b) const;
    };
    
    /**
        Compute distances for each pair of correspondences.
    */
    template<class Functor>
    Eigen::VectorXd distances(const Eigen::Matrix3d &f, Eigen::Ref<const Eigen::MatrixXd> a, Eigen::Ref<const Eigen::MatrixXd> b, Functor err) {
        Eigen::VectorXd errs(a.cols());
        for (Eigen::DenseIndex i = 0; i < a.cols(); ++i) {
            errs(i) = err(f, a.col(i), b.col(i));
        }
        return errs;
    }
    
}

#endif