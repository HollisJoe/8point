/**
    This file is part of 8point.

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#ifndef EIGHT_ESSENTIAL_H
#define EIGHT_ESSENTIAL_H

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace eight {
    
    /**
        Essential matrix from camera instrinsics and fundamental matrix.
    */
    Eigen::Matrix3d essentialMatrix(const Eigen::Matrix3d &k, const Eigen::Matrix3d &f);
    
    
    /**
        Recover pose from essential matrix up to scale.
    */
    Eigen::Matrix<double, 3, 4> pose(const Eigen::Matrix3d &e, const Eigen::Matrix3d &k, Eigen::Ref<const Eigen::MatrixXd> a, Eigen::Ref<const Eigen::MatrixXd> b);
    
}

#endif