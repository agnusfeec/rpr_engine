#include "fvector.h"

fvector::fvector()
{

}

VlGMM * fvector::codeBook(float *data, int numData, int dimension, int numComponents){
    // create a GMM object and cluster input data to get means, covariances
    // and priors of the estimated mixture
    VlGMM *gmm = vl_gmm_new (VL_TYPE_FLOAT, dimension, numComponents);
    vl_gmm_cluster (gmm, data, numData);

    return gmm;
}

float * fvector::encode(VlGMM * gmm, float *dataToEncode, int numDataToEncode){

    float * means ;
    float * covariances ;
    float * priors ;
    //float * posteriors ;
    float * enc;

    int numComponents = vl_gmm_get_num_clusters(gmm);
    int dimension = vl_gmm_get_dimension(gmm);

    std::cout << numComponents << " " << dimension << "\n";

    // allocate space for the encoding
    enc = (float *)vl_malloc(sizeof(float) * 2 * dimension * numComponents);

    // run fisher encoding
    means = (float *)vl_gmm_get_means(gmm);
    covariances = (float *)vl_gmm_get_covariances(gmm);
    priors = (float *)vl_gmm_get_priors(gmm);

    vl_size n = vl_fisher_encode
        (enc, VL_TYPE_FLOAT,
         means, dimension, numComponents,
         covariances,
         priors,
         dataToEncode, numDataToEncode,
         VL_FISHER_FLAG_IMPROVED
         );

    std::cout << n << "\n";

    return enc;
}

void fvector::gmm_write(VlGMM *gmm){

    // means and covariances have dimension rows and numCluster columns
    // priors is a vector of size numCluster
    //
    //float * posteriors ;
    float * means = (float *)vl_gmm_get_means(gmm);
    float * covariances = (float *)vl_gmm_get_covariances(gmm);
    float * priors  = (float *)vl_gmm_get_priors(gmm);
    int numCluster = vl_gmm_get_num_clusters(gmm);
    int dimension = vl_gmm_get_dimension(gmm);

    std::ofstream myfile ("gmm.txt");
    if (myfile.is_open())
    {
        myfile << dimension << " " << numCluster << "\n";

        int i=0;
        myfile << "#means\n";
        for(i=0; i<(dimension*numCluster)-1;i++)
            myfile << means[i] << " ";
        myfile << means[i] << "\n";

        myfile << "#covariances\n";
        for(i=0; i<(dimension*numCluster)-1;i++)
            myfile << covariances[i] << " ";
        myfile << covariances[i] << "\n";

        myfile << "#priors\n";
        for(i=0; i<(numCluster)-1;i++)
            myfile << priors[i] << " ";
        myfile << priors[i] << "\n";

        myfile.close();
      }
}

VlGMM * fvector::gmm_load(){

    // means and covariances have dimension rows and numCluster columns
    // priors is a vector of size numCluster
    //
    //float * posteriors ;
    VlGMM *gmm = NULL;
    float * means;
    float * covariances;
    float * priors;
    int numCluster;
    int dimension;

    std::string aux;

    std::ifstream myfile ("gmm.txt");
    if (myfile.is_open())
    {
        myfile >> dimension >> numCluster;

        gmm = vl_gmm_new (VL_TYPE_FLOAT, dimension, numCluster);

        int i=0;
        myfile >> aux; // # means
        means = (float *)vl_malloc(sizeof(float) * dimension * numCluster);
        for(i=0; i<(dimension*numCluster);i++) {
            myfile >> means[i];
            std::cout << means[i] << " ";
        }
        std::cout << "\n";
        vl_gmm_set_means(gmm, means);

        myfile >> aux;  //# covariances
        covariances = (float *)vl_malloc(sizeof(float) * dimension * numCluster);
        for(i=0; i<(dimension*numCluster);i++) {
            myfile >> covariances[i];
            std::cout << covariances[i] << " ";
        }
        std::cout << "\n";
        vl_gmm_set_covariances(gmm, covariances);

        priors = (float *)vl_malloc(sizeof(float) * dimension);
        myfile >> aux;  //# priors\n";
        for(i=0; i<(numCluster);i++) {
            myfile >> priors[i];
            std::cout << priors[i] << " ";
        }
        std::cout << "\n";
        vl_gmm_set_priors(gmm, priors);

        myfile.close();
    }

    return gmm;
}
