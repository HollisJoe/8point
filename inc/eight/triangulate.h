/**
    This file is part of 8point.

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#ifndef EIGHT_TRIANGULATE_H
#define EIGHT_TRIANGULATE_H

#include <Eigen/Core>

namespace eight {
    
    /** Triangulate rays using linear least squares.
     
        Based on
        Hartley, Richard I., and Peter Sturm. "Triangulation."
        Computer vision and image understanding 68.2 (1997): 146-157.
     */
    inline Eigen::Vector3d triangulate(const Eigen::Matrix<double, 3, 4> &cam0,
                                       const Eigen::Matrix<double, 3, 4> &cam1,
                                       const Eigen::Vector2d &u0,
                                       const Eigen::Vector2d &u1)
    {
        // Build 4x3 A and 4x1 b in a single 4x4
        Eigen::Matrix<double, 4, 4> X;
        X.row(0) = u0(0) * cam0.row(2) - cam0.row(0);
        X.row(1) = u0(1) * cam0.row(2) - cam0.row(1);
        X.row(2) = u1(0) * cam1.row(2) - cam1.row(0);
        X.row(3) = u1(1) * cam1.row(2) - cam1.row(1);
        
        return X.block<4,3>(0,0).jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(-X.col(3));
    }
    
    /** Triangulate rays using linear least squares.
     
        Based on
        Hartley, Richard I., and Peter Sturm. "Triangulation."
        Computer vision and image understanding 68.2 (1997): 146-157.
     */
    inline Eigen::Matrix<double, 3, Eigen::Dynamic> triangulateMany(const Eigen::Matrix<double, 3, 4> &cam0,
                                                                    const Eigen::Matrix<double, 3, 4> &cam1,
                                                                    Eigen::Ref<const Eigen::MatrixXd> u0,
                                                                    Eigen::Ref<const Eigen::MatrixXd> u1)
    {
        Eigen::Matrix<double, 3, Eigen::Dynamic> vecs(3, u0.cols());
        
        Eigen::Matrix<double, 4, 4> X;
        for (Eigen::DenseIndex i  = 0; i < u0.cols(); ++i) {
            // Build 4x3 A and 4x1 b in a single 4x4
            X.row(0) = u0(0, i) * cam0.row(2) - cam0.row(0);
            X.row(1) = u0(1, i) * cam0.row(2) - cam0.row(1);
            X.row(2) = u1(0, i) * cam1.row(2) - cam1.row(0);
            X.row(3) = u1(1, i) * cam1.row(2) - cam1.row(1);
            vecs.col(i) = X.block<4,3>(0,0).jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(-X.col(3));
        }
        
        return vecs;
    }
    
}

#endif