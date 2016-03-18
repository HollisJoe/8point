/**
 This file is part of 8point.
 
 Copyright(C) 2015/2016 Christoph Heindl
 All rights reserved.
 
 This software may be modified and distributed under the terms
 of the BSD license.See the LICENSE file for details.
*/

#include <eight/structure.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace eight {
    
    inline Eigen::Matrix<double, 3, 3> hat(Eigen::Ref<const Eigen::Vector3d> u) {
        
        Eigen::Matrix<double, 3, 3> h(3,3);
        h <<
            0.0, -u.z(), u.y(),
            u.z(), 0.0, -u.x(),
            -u.y(), u.x(), 0.0;
        
        return h;
    }

    Eigen::Matrix<double, 3, Eigen::Dynamic> structureFromTwoViews(const Eigen::Matrix<double, 3, 3> &k,
                                                                   const Eigen::Matrix<double, 3, 4> &pose1,
                                                                   Eigen::Ref<const Eigen::MatrixXd> u0,
                                                                   Eigen::Ref<const Eigen::MatrixXd> u1)
    {
        eigen_assert(u0.cols() == u1.cols());
        eigen_assert(u0.cols() > 0);
        
        
        Eigen::Matrix<double, 3, Eigen::Dynamic> points(3, u0.cols());
        
        // Note the system of equations is modified to only have depths of u0 and the scale factor as unknowns.
        
        Eigen::MatrixXd m = Eigen::MatrixXd::Zero(3 * u0.cols(), u0.cols() + 1);
        
        auto u0h = u0.colwise().homogeneous();
        auto u1h = u1.colwise().homogeneous();
        
        Eigen::Matrix<double, 3, 3> kinv = k.inverse();
        
        // Note, we need the inverse pose because of the way the linear systme is build.
        // Also note that the following equations do not work on pixel coordinates but normalized image coordinates.
        //
        //      d1 * u1 = d0 * R * u0 + yT.
        // Multiply by hat(u1) from the left, erases the first term (cross product with itself)
        //      hat(u1) * d1 * u1 = d0 * hat(u1) * R * u0 + y * hat(u1) * T
        //      0 = d0 * hat(u1) * R * u0 + y * hat(u1) * T
        // Here y and di are the unknown depths for the first set of normalized image coordinates.
        
        Eigen::Matrix<double, 3, 4> poseinv = Eigen::AffineCompact3d(pose1).inverse(Eigen::Isometry).matrix();
        Eigen::Matrix<double, 3, 3> r = poseinv.block<3,3>(0, 0);
        Eigen::Vector3d t = poseinv.block<3,1>(0, 3);
        
        for (Eigen::DenseIndex i  = 0; i < u0.cols(); ++i) {
            Eigen::Matrix<double, 3, 3> u1hat = hat(kinv * u1h.col(i));
            
            // Rotational part
            m.block<3,1>(i*3, i) = u1hat * r * kinv * u0h.col(i);
            
            // Translational part
            m.block<3,1>(i*3, m.cols()-1) = u1hat * t;
        }
        
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> e;
        e.compute(m.transpose() * m);
        Eigen::VectorXd x = e.eigenvectors().col(0);
        
        for (Eigen::DenseIndex i  = 0; i < u0.cols(); ++i) {
            points.col(i) = x(i) * kinv * u0h.col(i);
        }
        
        
        return points;
    }
    
}