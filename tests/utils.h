/**
 This file is part of 8point.
 
 Copyright(C) 2015/2016 Christoph Heindl
 All rights reserved.
 
 This software may be modified and distributed under the terms
 of the BSD license.See the LICENSE file for details.
 */

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace eight {
    namespace utils {
        
        inline Eigen::Matrix<double, 3, Eigen::Dynamic> samplePointsInBox(const Eigen::Vector3d &minCorner, const Eigen::Vector3d &maxCorner, Eigen::DenseIndex count) {
            Eigen::AlignedBox3d box(minCorner, maxCorner);
            Eigen::Matrix<double, 3, Eigen::Dynamic> points(3, count);
            for (int i = 0; i < count; ++i) {
                points.col(i) = box.sample();
            }
            return points;
        }
        
        
        inline Eigen::Matrix<double, 2, Eigen::Dynamic> projectPoints(const Eigen::Matrix3d &k, const Eigen::AffineCompact3d &t, const Eigen::Matrix<double, 3, Eigen::Dynamic> &points)
        {
            Eigen::Matrix<double, 3, 4> p = k * t.matrix();
            return (p * points.colwise().homogeneous()).colwise().hnormalized();
        }
    }
}