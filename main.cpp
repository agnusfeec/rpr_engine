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

float * vl_fv(float *data, int numData, int dimension, int numComponents, float *dataToEncode, int numDataToEncode){

    float * means ;
    float * covariances ;
    float * priors ;
    //float * posteriors ;
    float * enc;

    // create a GMM object and cluster input data to get means, covariances
    // and priors of the estimated mixture
    VlGMM *gmm = vl_gmm_new (VL_TYPE_FLOAT, dimension, numComponents);
    vl_gmm_cluster (gmm, data, numData);

    // allocate space for the encoding
    enc = (float *)vl_malloc(sizeof(float) * 2 * dimension * numComponents);

    // run fisher encoding
    means = (float *)vl_gmm_get_means(gmm);
    covariances = (float *)vl_gmm_get_covariances(gmm);
    priors = (float *)vl_gmm_get_priors(gmm);

    vl_fisher_encode
        (enc, VL_TYPE_FLOAT,
         means, dimension, numComponents,
         covariances,
         priors,
         dataToEncode, numDataToEncode,
         VL_FISHER_FLAG_IMPROVED
         );

    return enc;
}

// matrices in VLFeat are stored in memory in column major order.
// http://www.vlfeat.org/api/conventions.html#conventions-storage
float * mat2vec(Mat m){
    int i = 0;
    float * d = (float *)malloc(m.rows*m.cols*sizeof(float));
    // testar sucesso da alocação

    for(int c=0; c<m.cols; c++){
        for(int r=0; r<m.rows; r++){
            d[i++] = m.at<float>(r,c);
        }
    }

    return d;
}

int kp_write(std::vector<KeyPoint> keypoints){
     std::ofstream myfile ("keypoints.csv");
     if (myfile.is_open())
     {
       myfile << "kaze " << keypoints.size() << "\n";

       for(KeyPoint kp : keypoints) {
           // float x, float y, float _size, float _angle=-1, float _response=0, int _octave=0, int _class_id=-1
           myfile << kp.pt.x << " " << kp.pt.y << " " << kp.size << " " << kp.angle << " ";
           myfile << kp.response << " " << kp.octave << " " << kp.class_id << "\n";
       }
       myfile.close();
     }
     return 0;
}

int ds_write(Mat descriptors){
     std::ofstream myfile ("descriptors.csv");
     if (myfile.is_open())
     {
       myfile << "kaze " << descriptors.rows << " " << descriptors.cols << "\n";

       for(int r=0; r<descriptors.rows; r++){
           for(int c=0; c<descriptors.cols-1; c++){
               myfile << descriptors.at<float>(r,c) << " ";
           }
           myfile << descriptors.at<float>(r,descriptors.cols) << "\n";
       }

       myfile.close();
     }
     return 0;
}

int main(int argc, char *argv[])
{
    //QCoreApplication a(argc, argv);

    //return a.exec();

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

    kp_write(keypoints1);
    ds_write(descriptors1);

    float * enc = vl_fv(mat2vec(descriptors1), descriptors1.rows*descriptors1.cols, 1,
          descriptors1.cols, mat2vec(descriptors2), descriptors2.rows*descriptors2.cols);

    for(int i=0; i<128; i++) {
        std::cout << enc[i] << " ";
    }
    std::cout << "\n";
    enc = vl_fv(mat2vec(descriptors1), descriptors1.rows*descriptors1.cols, 1,
              descriptors1.cols, mat2vec(descriptors1), descriptors1.rows*descriptors1.cols);

    for(int i=0; i<128; i++) {
        std::cout << enc[i] << " ";
    }

    std::cout << "\n";

    waitKey();
    return 0;
}
