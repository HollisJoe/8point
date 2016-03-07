/**
 This file is part of 8point.
 
 Copyright(C) 2015/2016 Christoph Heindl
 All rights reserved.
 
 This software may be modified and distributed under the terms
 of the BSD license.See the LICENSE file for details.
*/

#include <eight/project.h>
#include <Eigen/Geometry>

namespace eight {
    
    Eigen::Matrix<double, 3, 4> perspectiveProjectionMatrix(const Eigen::Matrix<double, 3, 3> &k,
                                                            const Eigen::Matrix<double, 3, 3> &r,
                                                            const Eigen::Vector3d &t)
    {
        Eigen::Isometry3d iso;
        iso.linear() = r;
        iso.translation() = t;
        
        return k * iso.matrix().block<3,4>(0,0);
    }
    
    Eigen::Matrix<double, 3, Eigen::Dynamic> perspectiveProject(Eigen::Ref<const Eigen::MatrixXd> points, const Eigen::Matrix<double, 3, 4>  &p) {
        return p * points.colwise().homogeneous();
    }
    
    Eigen::Matrix<double, 3, Eigen::Dynamic> perspectiveProject(Eigen::Ref<const Eigen::MatrixXd> points,
                                                                const Eigen::Matrix<double, 3, 3> &k,
                                                                const Eigen::Matrix<double, 3, 3> &r,
                                                                const Eigen::Vector3d &t)
    {
        return perspectiveProject(points, perspectiveProjectionMatrix(k, r, t));
    }
    
    Eigen::Matrix<double, 3, Eigen::Dynamic> perspectiveProject(Eigen::Ref<const Eigen::MatrixXd> points,
                                                                const Eigen::Matrix<double, 3, 3> &k,
                                                                const Eigen::Matrix<double, 3, 4> &g)
    {
        return perspectiveProject(points, k * g);
    }
    
    
}