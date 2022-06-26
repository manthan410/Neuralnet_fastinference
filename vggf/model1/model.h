#pragma once 

namespace FAST_INFERENCE {    
    #define REF_ACCURACY 74.81

    #define N_CLASSES 10
    #define N_FEATURES 784

    void predict_model(double const * const x, double * pred);
}