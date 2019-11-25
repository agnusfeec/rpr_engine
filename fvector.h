#ifndef FVECTOR_H
#define FVECTOR_H

extern "C" {
  #include <vl/generic.h>
  #include <vl/fisher.h>
  #include <vl/gmm.h>
  #include <vl/mathop.h>
  #include <vl/kmeans.h>
}

#include <string.h>
#include <iostream>
#include <fstream>

class fvector
{
public:
    fvector();
    VlGMM * codeBook(float *data, int numData, int dimension, int numComponents);
    float * encode(VlGMM * gmm, float *dataToEncode, int numDataToEncode);
    void gmm_write(VlGMM *gmm);
    VlGMM * gmm_load();
};

#endif // FV_H
