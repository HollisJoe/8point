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
    /*Eigen::Matrix<typename Eigen::EigenBase<Derived>::Scalar, typename Eigen::EigenBase<Derived>::*/ selectColumnsByIndex(const EigenBase<Derived> &m, IndexIterator begin, IndexIterator end) {
        typename Matrix::Index count = (typename Matrix::Index)std::distance(begin, end);

        Matrix r(m.rows(), count);
        typename Matrix::Index i = 0;
        while (begin != end) {
            r.col(i) = m.col(*begin);
            ++begin;
            ++i;
        }

        return r;
    }
    
}

#endif