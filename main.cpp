#include <QCoreApplication>
#include <iostream>
#include <fstream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

// vlfeat
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
  #include <vl/generic.h>
  #include <vl/fisher.h>
  #include <vl/gmm.h>
  #include <vl/mathop.h>
  #include <vl/kmeans.h>
}

#include "feature.h"
#include "fvector.h"
#include "util.h"

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;

const char* keys =
    "{ help h   |                  | Print help message. }"
    "{ input1 i1| tattoo-god.jpg   | Path to input image 1. }"
    "{ input2 i2| tatoo_01.jpg     | Path to input image 2. }";

int vl_test(){
    VL_PRINT ("Hello world!\n") ;
    return 0;
}

int main(int argc, char *argv[])
{
    //QCoreApplication a(argc, argv);

    //return a.exec();

    feature ft = feature();
    fvector fisher = fvector();

    vl_test();

    CommandLineParser parser( argc, argv, keys );

    std::cout << parser.get<String>("input1") << std::endl;
    std::cout << parser.get<String>("input2") << std::endl;

    Mat imga = imread( parser.get<String>("input1"), IMREAD_GRAYSCALE );
    Mat img2 = imread( parser.get<String>("input2"), IMREAD_GRAYSCALE );
    if ( imga.empty() || img2.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        parser.printMessage();
        return -1;
    }

    Rect roi = Rect(128, 140, 162, 182);
    //Rect roi = Rect(0, 0, imga.cols, imga.rows);

    Mat img1 = Mat(imga, roi);

    //-- Step 1: Detect the keypoints using KAZE Detector, compute the descriptors
    bool 	extended = false;
    bool 	upright = false;
    float 	threshold = 0.001f;
    int 	nOctaves = 4;
    int 	nOctaveLayers = 4;
    int 	diffusivity = KAZE::DIFF_PM_G2;

    Ptr<KAZE> detector = KAZE::create( extended, upright, threshold, nOctaves, nOctaveLayers, diffusivity );
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute( img1, noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( img2, noArray(), keypoints2, descriptors2 );

    //-- Step 2: Matching descriptor vectors with a Flann Based matcher
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< DMatch > matches;
    matcher->match( descriptors1, descriptors2, matches );

    std::vector< DMatch > matches_aux;
//    for(unsigned int i=0; i<matches.size() ; i++) {
//        std::cout << matches[i].distance << std::endl;
//        if(matches[i].distance<0.2) matches_aux.push_back(matches[i]);
//    }

    for(auto match : matches) {
        //std::cout << match.distance << std::endl;
        if(match.distance<0.3) matches_aux.push_back(match);
    }

    //-- Draw matches
    Mat img_matches;
    drawMatches( img1, keypoints1, img2, keypoints2, matches_aux, img_matches );
    //-- Show detected matches
    imshow("Matches", img_matches );

    ft.kp_write(keypoints1);
    ft.ds_write(descriptors1);

    VlGMM * gmm = fisher.codeBook(mat2vec(descriptors1), descriptors1.rows, descriptors1.cols,1);
    float * enc = fisher.encode(gmm, mat2vec(descriptors2), descriptors2.rows);

    for(int i=0; i<128; i++) {
        std::cout << enc[i] << " ";
    }
    std::cout << "\n\n";
    enc = fisher.encode(gmm, mat2vec(descriptors1), descriptors1.rows);

    for(int i=0; i<128; i++) {
        std::cout << enc[i] << " ";
    }

    std::cout << "\n";

    descriptors1 = ft.ds_load();

    fisher.gmm_write(gmm);

    gmm = fisher.gmm_load();

    waitKey();
    return 0;
}
