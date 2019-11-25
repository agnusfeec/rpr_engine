#ifndef FEATURE_H
#define FEATURE_H

#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::xfeatures2d;

class feature
{
public:
    feature();
    int kp_write(std::vector<KeyPoint> keypoints);
    int ds_write(Mat descriptors);
    Mat ds_load();
};

#endif // FEATURE_H
