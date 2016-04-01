/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#include <eight/fundamental.h>
#include <eight/normalize.h>
#include <eight/distance.h>
#include <eight/essential.h>
#include <eight/project.h>
#include <eight/select.h>
#include <eight/triangulate.h>
#include <random>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <fstream>
#include <random>

#include <Eigen/Sparse>

#ifdef _WIN32
#define GOOGLE_GLOG_DLL_DECL
#pragma warning(push)
#pragma warning(disable : 4251 4355)
#include <ceres/ceres.h>
#include "ceres/rotation.h"
#include <glog/logging.h>
#pragma warning(pop)
#else
#include <ceres/ceres.h>
#include "ceres/rotation.h"
#include <glog/logging.h>
#endif

#include "../tests/utils.h"



class KLT {
public:

    KLT(const cv::Mat &img, const std::vector<cv::Point2f> &initial)
        :_next(initial)
    {
        cv::cvtColor(img, _nextGray, CV_BGR2GRAY);
        _nextStatus.resize(initial.size(), 255);
    }

    void update(cv::Mat &img) {        
        
        std::swap(_prev, _next);
        std::swap(_prevGray, _nextGray);
        std::swap(_prevStatus, _nextStatus);

        cv::cvtColor(img, _nextGray, CV_BGR2GRAY);
        cv::TermCriteria term(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 50, 0.001);
        cv::calcOpticalFlowPyrLK(_prevGray, _nextGray, _prev, _next, _nextStatus, _err, cv::Size(21, 21), 5, term);

        for (size_t i = 0; i < _nextStatus.size(); ++i) {
            _nextStatus[i] &= _prevStatus[i];
            _nextStatus[i] &= (_err[i] < 5);
        }
    }

    std::vector<cv::Point2f> &location() {
        return _next;
    }

    std::vector<uchar> &status() {
        return _nextStatus;
    }


private:
    std::vector<cv::Point2f> _prev, _next;
    cv::Mat _prevGray, _nextGray;
    std::vector<uchar> _prevStatus, _nextStatus;
    std::vector<float> _err;
};

Eigen::MatrixXd toEight(const std::vector<cv::Point2f> &x) {
    Eigen::MatrixXd m(2, x.size());

    for (size_t i = 0; i < x.size(); ++i) {
        m(0, i) = x[i].x;
        m(1, i) = x[i].y;
    }

    return m;
}

Eigen::MatrixXd imagePointsToRetinaPoints(const std::vector<cv::Point2f> &p, const Eigen::Matrix3d &kinv)
{
    Eigen::MatrixXd r(3, p.size());

    for (size_t i = 0; i < p.size(); ++i) {
        Eigen::Vector3d x(p[i].x, p[i].y, 1.0);
        r.col(i) = kinv * x;
    }

    return r;
}

void findFeaturesInReference(cv::Mat &gray, std::vector<cv::Point2f> &corners) {
    double qualityLevel = 0.02;
    double minDistance = 10;
    int blockSize = 5;
    bool useHarrisDetector = false;
    double k = 0.04;
    int maxCorners = 2000;

    cv::goodFeaturesToTrack(gray,
        corners,
        maxCorners,
        qualityLevel,
        minDistance,
        cv::Mat(),
        blockSize,
        useHarrisDetector,
        k);
    
    cv::TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03);
    cv::cornerSubPix(gray, corners, cv::Size(10,10), cv::Size(-1,-1), termcrit);
}

void trackFeatures(cv::Mat ref, const std::vector<cv::Point2f> &refLocs, cv::Mat target, std::vector<cv::Point2f> &targetLocs, std::vector<uchar> &status)
{
    KLT klt(ref, refLocs);
    klt.update(target);
    targetLocs = klt.location();
    status = klt.status();
}

struct GeneralReprojectionError {
    GeneralReprojectionError(const double *no, const double *np)
        : _no(no), _np(np)
    {}

    template <typename T>
    bool operator()(
        const T* const camera,
        const T* const depth,
        T* residuals) const 
    {
        T p[3] = {T(_np[0]) / depth[0], T(_np[1]) / depth[0], T(1.0) / depth[0] };

        T rpt[3] = {
            p[0] - camera[2] * p[1] + camera[1] * p[2] + camera[3],
            p[0] * camera[2] + p[1] - camera[0] * p[2] + camera[4],
            -p[0] * camera[1] + p[1] * camera[0] + p[2] + camera[5]
        };
        
        residuals[0] = T(_no[0]) - rpt[0] / rpt[2] ;
        residuals[1] = T(_no[1]) - rpt[1] / rpt[2] ;

        return true;
    }

    static ceres::CostFunction* Create(const double *no, const double *np)
    {
        return (new ceres::AutoDiffCostFunction<GeneralReprojectionError, 2, 6, 1>(
            new GeneralReprojectionError(no, np)));
    }

    const double *_no;
    const double *_np;
};

void normalizePerspective(Eigen::Matrix<double, 3, Eigen::Dynamic> &x) {
    for (Eigen::DenseIndex i = 0; i < x.cols(); ++i) {
        x.col(i) /= x.col(i).z();
    }
}

void writePly(const char *path, const Eigen::MatrixXd &points, const Eigen::MatrixXd &colors, const std::vector<uchar> &status) {
    std::ofstream ofs(path);
    
    size_t valid = std::count(status.begin(), status.end(), 1);
    
    ofs
        << "ply" << std::endl
        << "format ascii 1.0" << std::endl
        << "element vertex " << valid << std::endl
        << "property float x" << std::endl
        << "property float y" << std::endl
        << "property float z" << std::endl
        << "property uchar red" << std::endl
        << "property uchar green" << std::endl
        << "property uchar blue" << std::endl
        << "end_header" << std::endl;
    
    
    for (Eigen::DenseIndex i = 0; i < points.cols(); ++i) {
        if (!status[i])
            continue;
        
        Eigen::Vector3d x = points.col(i);
        Eigen::Vector3d c = colors.col(i);
        
        ofs << x(0) << " " << x(1) << " " << x(2) << " " << (int)c(0) << " " << (int)c(1) << " " << (int)c(2) << std::endl;
    }
    
    ofs.close();
}

void dense(cv::Mat &depths, cv::Mat &colors) {
    // Variational with energy: (dout - dsparse)^2 + lambda * |nabla(dout)|^2
    // First term is only defined for sparse pixel positions.
    // leads to linear system of equations: Ax = b with one row per pixel: 
    //  dout(x,y) + lambda * laplacian(dout(x,y)) = dsparse(x,y)
    //  dout(x,y) + lambda * (-4*dout(x,y) + dout(x,y-1) + dout(x+1,y) + dout(x,y+1) + dout(x-1,y)) = dsparse(x,y)


    typedef Eigen::Triplet<double> T;
    std::vector<T> triplets;

    std::cout << __LINE__ << std::endl;

    int rows = depths.rows;
    int cols = depths.cols;

    Eigen::MatrixXd rhs(rows * cols, 1);
    rhs.setZero();

    std::cout << __LINE__ << std::endl;

    const double lambda = 0.1;

    int idx = 0;
    for (int y = 0; y < depths.rows; ++y) {
        for (int x = 0; x < depths.cols; ++x, ++idx) {

            double d = depths.at<double>(y, x);
            
            double c = 0.0;

            if (d > 0.0) {
                rhs(idx, 0) = d;
                c += 1.0;
            }
            

            if (y > 0) {
                // North neighbor 
                c -= lambda;
                triplets.push_back(T(idx, (y - 1)*cols + x, lambda));
            }

            if (x > 0) {
                // West neighbor 
                c -= lambda;
                triplets.push_back(T(idx, (y)*cols + (x-1), lambda));
            }

            if (y < (rows - 1)) {
                // South neighbor 
                c -= lambda;
                triplets.push_back(T(idx, (y+1)*cols + x, lambda));
            }

            if (x < (cols - 1)) {
                // East neighbor 
                c -= lambda;
                triplets.push_back(T(idx, (y)*cols + (x+1), lambda));
            }

            // Center
            triplets.push_back(T(idx, idx, c));
        }
    }

    std::cout << __LINE__ << std::endl;

    Eigen::SparseMatrix<double> A(rows*cols, rows*cols);
    A.setFromTriplets(triplets.begin(), triplets.end());

    std::cout << __LINE__ << std::endl;

    Eigen::SparseLU< Eigen::SparseMatrix<double> > solver;
    solver.analyzePattern(A);
    solver.factorize(A);

    Eigen::MatrixXd result(rows*cols, 1);
    result = solver.solve(rhs);

    idx = 0;
    for (int y = 0; y < depths.rows; ++y) {
        for (int x = 0; x < depths.cols; ++x, ++idx) {
            depths.at<double>(y, x) = result(idx, 0);
        }
    }

}

int main(int argc, char **argv) {

    google::InitGoogleLogging(argv[0]);

    if (argc != 2) {
        std::cerr << argv[0] << " videofile" << std::endl;
        return -1;
    }

    cv::VideoCapture vc;
    if (!vc.open(argv[1])) {
        std::cerr << "Failed to open video" << std::endl;
        return -1;
    }

    // Taken from http://yf.io/p/tiny/
    // Use 8point_defocus.exe "stream\stone6_still_%04d.png"
    
    Eigen::Matrix3d k;
    k <<
        1781.0, 0.0, 960.0,
        0.0, 1781.0, 540.0,
        0.0, 0.0, 1.0;

    Eigen::Matrix3d invk = k.inverse();


    // Detect trackable features in reference frame
    cv::Mat ref, refGray;
    vc >> ref;
    cv::cvtColor(ref, refGray, CV_BGR2GRAY);

    std::vector< Eigen::MatrixXd > retinapoints;
    std::vector<cv::Point2f> corners;
    
    findFeaturesInReference(refGray, corners);
    std::vector<uchar> status(corners.size(), 1);
    
    retinapoints.push_back(imagePointsToRetinaPoints(corners, invk));


    cv::Mat f;
    while (vc.grab()) {
        vc.retrieve(f);

        std::vector<cv::Point2f> loc;
        std::vector<uchar> s;

        trackFeatures(ref, corners, f, loc, s);
        
        retinapoints.push_back(imagePointsToRetinaPoints(loc, invk));
        for (size_t i = 0; i < s.size(); ++i) {
            status[i] &= s[i];

            if (status[i]) {
                cv::circle(f, loc[i], 2, cv::Scalar(0, 255, 0));
            }
        }
        
        cv::Mat tmp;
        cv::resize(f, tmp, cv::Size(), 0.5, 0.5);
        cv::imshow("track", tmp);
        cv::waitKey(10);
    }
    
    retinapoints[0].row(2).setRandom(); // Initial inverse depth
    retinapoints[0].row(2).array() += 1.0;

    // Setup camera parameters
    Eigen::MatrixXd camparams(6, retinapoints.size());
    camparams.setZero();

    // Setup nnls system
    ceres::Problem problem;

    // For each camera
    for (size_t f = 1; f < retinapoints.size(); ++f) {
        // For each point observed
        for (Eigen::DenseIndex p = 0; p < retinapoints[f].cols(); ++p) {
            if (!status[p])
                continue;

            ceres::CostFunction* cost_function = GeneralReprojectionError::Create(
                retinapoints[f].col(p).data(),
                retinapoints[0].col(p).data()
            );
            

            problem.AddResidualBlock(
                cost_function,
                NULL /* squared loss */,
                camparams.col(f).data(),
                retinapoints[0].col(p).data() + 2);
        }
    }


    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 400;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    
    cv::Mat depths(ref.size(), CV_64FC1);
    depths.setTo(0);

    Eigen::MatrixXd colors(3, retinapoints[0].cols());
    Eigen::MatrixXd points(3, retinapoints[0].cols());
    Eigen::DenseIndex i = 0;

    for (Eigen::DenseIndex i = 0; i < retinapoints[0].cols(); ++i) {           
        if (!status[i])
            continue;

        Eigen::Vector3d q = retinapoints[0].col(i);
        q.x() /= q.z();
        q.y() /= q.z();
        q.z() = 1.0 / q.z();

        points.col(i) = q;

        cv::Vec3b c = ref.at<cv::Vec3b>(corners[i]);
        colors(0, i) = c(2);
        colors(1, i) = c(1);
        colors(2, i) = c(0);

            
        depths.at<double>(corners[i]) = q.z();
    }
    writePly("points.ply", points, colors, status);

    dense(depths, ref);    

    double minv, maxv;
    cv::minMaxLoc(depths, &minv, &maxv);

    cv::Mat tmp;
    depths.convertTo(tmp, CV_8U, 255.0 / (maxv - minv), -minv * 255.0 / (maxv - minv));
    cv::resize(tmp, tmp, cv::Size(), 0.5, 0.5);
    cv::imshow("dense", tmp);
    cv::waitKey();


    //cv::FileStorage file("sparse.yml", cv::FileStorage::WRITE);
    //file << ref;
    //file << depths;

}
