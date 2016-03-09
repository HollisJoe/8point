/**
    This file is part of 8point.

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#ifndef EIGHT_SELECT_H
#define EIGHT_SELECT_H

#include <Eigen/Core>
#include <iterator>

namespace eight {
    
    template<class Derived, class IndexIterator>
    Eigen::Matrix<
        typename Eigen::MatrixBase<Derived>::Scalar,
        Eigen::MatrixBase<Derived>::RowsAtCompileTime,
        Eigen::MatrixBase<Derived>::ColsAtCompileTime
    >
    selectColumnsByIndex(const Eigen::MatrixBase<Derived> &m, IndexIterator begin, IndexIterator end) {
        
        Eigen::DenseIndex count = (Eigen::DenseIndex)std::distance(begin, end);
        
        Eigen::Matrix<
            typename Eigen::MatrixBase<Derived>::Scalar,
            Eigen::MatrixBase<Derived>::RowsAtCompileTime,
            Eigen::MatrixBase<Derived>::ColsAtCompileTime
        > r(m.rows(), count);
        
        
        Eigen::DenseIndex i = 0;
        while (begin != end) {
            r.col(i++) = m.col(*begin++);
        }

        return r;
    }
    
}

#endif