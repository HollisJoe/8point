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
#include <vector>

namespace eight {
    
    /**
        Estimate fundamental matrix from pairs of corresponding image points
    */
    Eigen::Matrix3d fundamentalMatrixUnnormalized(Eigen::Ref<const Eigen::MatrixXd> a, Eigen::Ref<const Eigen::MatrixXd> b);
    
    /**
        Estimate fundamental matrix from pairs of corresponding image points.

        All points are considered inliers.
     */
    Eigen::Matrix3d fundamentalMatrix(Eigen::Ref<const Eigen::MatrixXd> a, Eigen::Ref<const Eigen::MatrixXd> b);

    /**
        Estimate fundamental matrix from pairs of corresponding image points when outliers are present.

        \param a Image coordinates in first image.
        \param b Image coordinates in second image.
        \param inliers Inlier column indices of best solution.
        \param d Tolerated distance from the model for inliers (Sampson Error).
        \param e Assumed outlier percent in data set.
        \param p Probability that at least one valid set of inliers is chosen.
    */
    Eigen::Matrix3d fundamentalMatrixRobust(
        Eigen::Ref<const Eigen::MatrixXd> a, 
        Eigen::Ref<const Eigen::MatrixXd> b, 
        std::vector<Eigen::DenseIndex> &inliers, 
        double d = 3.0, double e = 0.2, double p = 0.99);
    
}

#endif