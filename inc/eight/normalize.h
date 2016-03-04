/**
    This file is part of 8point.

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#ifndef EIGHT_NORMALIZE_H
#define EIGHT_NORMALIZE_H

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace eight {
    
    /**
        Find normalizing transform of image points to support the conditioning of fundamental matrices.
    */
    Eigen::Affine2d findIsotropicNormalizingTransform(Eigen::Ref<const Eigen::MatrixXd> a);
    
}

#endif