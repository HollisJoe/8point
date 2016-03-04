/**
    This file is part of 8point.

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#ifndef EIGHT_FUNDAMENTAL_H
#define EIGHT_FUNDAMENTAL_H

#include <Eigen/Core>

namespace eight {
    
    /**
        Estimate fundamental matrix from pairs of corresponding image points
    */
    Eigen::Matrix3d findFundamentalMatrix(Eigen::Ref<const Eigen::MatrixXd> a, Eigen::Ref<const Eigen::MatrixXd> b);
    
    /**
        Estimate fundamental matrix from pairs of corresponding image points.
     */
    Eigen::Matrix3d findFundamentalMatrixNormalized(Eigen::Ref<const Eigen::MatrixXd> a, Eigen::Ref<const Eigen::MatrixXd> b);
    
}

#endif