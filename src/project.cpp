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

    Eigen::Matrix<double, 3, 4> cameraPose(const Eigen::Matrix<double, 3, 3> &r, const Eigen::Vector3d &t)
    {
        Eigen::AffineCompact3d iso;
        iso.linear() = r;
        iso.translation() = t;

        return iso.matrix();
    }
    
    Eigen::Matrix<double, 3, 4> cameraMatrix(const Eigen::Matrix<double, 3, 3> &k, const Eigen::Matrix<double, 3, 3> &r, const Eigen::Vector3d &t)
    {
        return cameraMatrix(k, cameraPose(r, t));
    }

    Eigen::Matrix<double, 3, 4> cameraMatrix(const Eigen::Matrix<double, 3, 3> &k, const Eigen::AffineCompact3d &pose)
    {
        return k * pose.inverse(Eigen::Isometry).matrix();
    }

    Eigen::Matrix<double, 3, 4> cameraMatrix(const Eigen::Matrix<double, 3, 3> &k,
                                             const Eigen::Matrix<double, 3, 4> &pose)
    {        
        Eigen::AffineCompact3d iso(pose);        
        return k * iso.inverse(Eigen::Isometry).matrix();
    }
    
    Eigen::Matrix<double, 3, Eigen::Dynamic> perspectiveProject(Eigen::Ref<const Eigen::MatrixXd> points, const Eigen::Matrix<double, 3, 4>  &p) {
        return p * points.colwise().homogeneous();
    }
    
    
}