/**
 This file is part of 8point.
 
 Copyright(C) 2015/2016 Christoph Heindl
 All rights reserved.
 
 This software may be modified and distributed under the terms
 of the BSD license.See the LICENSE file for details.
*/

#include <eight/normalize.h>

namespace eight {
    
    Eigen::Affine2d findIsotropicNormalizingTransform(Eigen::Ref<const Eigen::MatrixXd> a) {
        Eigen::Vector2d mean = a.rowwise().mean();
        Eigen::Vector2d stddev = (a.colwise() - mean).array().square().rowwise().mean().sqrt();
        
        Eigen::Affine2d t;
        t = Eigen::Scaling(1.0 / stddev.norm()) *  Eigen::Translation2d(-mean);
        return t;
    }
    
}