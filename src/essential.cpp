/**
 This file is part of 8point.
 
 Copyright(C) 2015/2016 Christoph Heindl
 All rights reserved.
 
 This software may be modified and distributed under the terms
 of the BSD license.See the LICENSE file for details.
*/

#include <eight/essential.h>
#include <eight/triangulate.h>
#include <Eigen/SVD>
#include <iostream>

namespace eight {
    
    Eigen::Matrix3d essentialMatrix(const Eigen::Matrix3d &k, const Eigen::Matrix3d &f) {
        return k.transpose() * f * k;
    }
    
    inline Eigen::Matrix4d toHomogeneousTransform(const Eigen::Matrix3d &rot, const Eigen::Vector3d &trans) {
        Eigen::Matrix4d t;
        t.setIdentity();
        t.block<3,3>(0,0) = rot;
        t.block<3,1>(0,3) = trans;
        return t;
    }
    
    Eigen::Affine3d pose(const Eigen::Matrix3d &e, const Eigen::Matrix3d &k, Eigen::Ref<const Eigen::MatrixXd> a, Eigen::Ref<const Eigen::MatrixXd> b) {
        
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(e, Eigen::ComputeThinU | Eigen::ComputeThinV);
        
        // Assuming the first camera at identity, there are four possible solutions that need to be tested for
        // the second camera.
        
        Eigen::Matrix3d u = svd.matrixU();
        Eigen::Matrix3d v = svd.matrixV();
        
        if (u.determinant() < 0.0)
            u *= -1.0;
        if (v.determinant() < 0.0)
            v *= -1.0;
        
        
        Eigen::Matrix3d w;
        w <<
            0.0, -1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 1.0;
        
        
        Eigen::Matrix3d r0 = u * w * v.transpose();
        Eigen::Matrix3d r1 = u * w.transpose() * v.transpose();
        Eigen::Vector3d t0 = u.col(2);
        Eigen::Vector3d t1 = -u.col(2);
        
        // Test possible solutions. According to Hartley testing one point for being infront of both cameras should be
        // enough.
        
        Eigen::Matrix4d k4 = Eigen::Matrix4d::Zero(4, 4);
        k4.block<3,3>(0,0) = k;
        
        Eigen::Matrix4d camFirst = k4;
        Eigen::Matrix4d camSecond[4] = {
            toHomogeneousTransform(r0, t0),
            toHomogeneousTransform(r0, t1),
            toHomogeneousTransform(r1, t0),
            toHomogeneousTransform(r1, t1)
        };
        
        for (int i = 0 ; i < 4; ++i) {
            Eigen::Vector3d p = triangulate(camFirst, k4 * camSecond[i], a.col(0), b.col(0));
            Eigen::Vector3d pp = (camSecond[i].inverse() * p.colwise().homogeneous()).colwise().hnormalized();
            
            if (p.z() >= 0.0 & pp.z() >= 0.0) {
                Eigen::Affine3d t;
                t.matrix() = camSecond[i];
                return t;
            }
        }
        
        return Eigen::Affine3d();
    }
    
}