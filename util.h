#ifndef UTIL_H
#define UTIL_H

// matrices in VLFeat are stored in memory in column major order.
// http://www.vlfeat.org/api/conventions.html#conventions-storage
float * mat2vec(Mat m){
    int i = 0;
    float * d = (float *)malloc(m.rows*m.cols*sizeof(float));
    // testar sucesso da alocação

    for(int c=0; c<m.cols; c++){
        for(int r=0; r<m.rows; r++){
            d[i++] = m.at<float>(r,c);
            //std::cout << "(" << d[i] << " " << m.at<float>(r,c) << ") ";
            //i++;
        }
    }

    return d;
}

#endif // UTIL_H
