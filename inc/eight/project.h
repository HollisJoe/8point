/**
    This file is part of 8point.

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#ifndef EIGHT_PROJECT_H
#define EIGHT_PROJECT_H

#include <Eigen/Core>

namespace eight {
    
    /**
        Assemble perspective projection matrix.
    */
    Eigen::Matrix<double, 3, 4> perspectiveProjectionMatrix(const Eigen::Matrix<double, 3, 3> &k,
                                                            const Eigen::Matrix<double, 3, 3> &r,
                                                            const Eigen::Vector3d &t);
    
    /**
        Perspective projection of three-dimensional points.
    */
    Eigen::Matrix<double, 3, Eigen::Dynamic> perspectiveProject(Eigen::Ref<const Eigen::MatrixXd> points, const Eigen::Matrix<double, 3, 4>  &p);
    
    
    /**
     Perspective projection of three-dimensional points.
     */
    Eigen::Matrix<double, 3, Eigen::Dynamic> perspectiveProject(Eigen::Ref<const Eigen::MatrixXd> points,
                                                                const Eigen::Matrix<double, 3, 3> &k,
                                                                const Eigen::Matrix<double, 3, 3> &r,
                                                                const Eigen::Vector3d &t);
    
    /**
     Perspective projection of three-dimensional points.
     */
    Eigen::Matrix<double, 3, Eigen::Dynamic> perspectiveProject(Eigen::Ref<const Eigen::MatrixXd> points,
                                                                const Eigen::Matrix<double, 3, 3> &k,
                                                                const Eigen::Matrix<double, 3, 4> &g);
    
}

#endif