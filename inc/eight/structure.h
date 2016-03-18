/**
    This file is part of 8point.

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#ifndef EIGHT_STRUCTURE_H
#define EIGHT_STRUCTURE_H

#include <Eigen/Core>

namespace eight {
    
    /** 
        Reconstruct 3d points from two views.
     
        Simultanously solves for all pixel correspondences using linear least squares.
     
        \param k Camera intrinsic matrix
        \param pose1 Translation and orientation of the second camera
        \param u0 pixel coordinates in the first image
        \param u1 pixel coordinates in the second image
     
        Ma, Yi, et al. 
        An invitation to 3-d vision: from images to geometric models.
        Vol. 26. Springer Science & Business Media, 2012. section 5.2.2
     
     */
    Eigen::Matrix<double, 3, Eigen::Dynamic> structureFromTwoViews(const Eigen::Matrix<double, 3, 3> &k,
                                                                   const Eigen::Matrix<double, 3, 4> &pose1,
                                                                   Eigen::Ref<const Eigen::MatrixXd> u0,
                                                                   Eigen::Ref<const Eigen::MatrixXd> u1);
    
}

#endif