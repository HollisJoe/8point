/**
 This file is part of 8point.
 
 Copyright(C) 2015/2016 Christoph Heindl
 All rights reserved.
 
 This software may be modified and distributed under the terms
 of the BSD license.See the LICENSE file for details.
*/

#include <eight/distance.h>

namespace eight {
    
    double SampsonDistanceSquared::operator()(const Eigen::Matrix3d &f, Eigen::Ref<const Eigen::Vector2d> a, Eigen::Ref<const Eigen::Vector2d> b) const {
        Eigen::Vector3d fa = f * a.homogeneous();
        Eigen::Vector3d fb = f.transpose() * b.homogeneous();
        
        double bfa = b.homogeneous().transpose() * fa;
        
        return (bfa * bfa) / (fa.topRows(2).squaredNorm() + fb.topRows(2).squaredNorm());
    }
    
    
}