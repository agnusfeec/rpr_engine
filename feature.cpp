#include "feature.h"

feature::feature()
{

}

int feature::kp_write(std::vector<KeyPoint> keypoints){
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

int feature::ds_write(Mat descriptors){
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

Mat feature::ds_load(){
    std::string tipo;
    unsigned int rows, cols;
    Mat descriptors;

    std::ifstream myfile ("descriptors.csv");
    if (myfile.is_open())
    {
       myfile >> tipo >> rows >> cols;

        descriptors = Mat(rows, cols, CV_32F);

        for(unsigned int r=0; r<rows; r++){
            for(unsigned int c=0; c<cols; c++){
                myfile >> descriptors.at<float>(Point(r,c));
            }
        }

        myfile.close();
    }

    return descriptors;
}
