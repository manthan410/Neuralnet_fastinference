#pragma once 

namespace FAST_INFERENCE {
    #define N_CLASSES 1000
    #define N_FEATURES 3

    void predict_model(double const * const x, double * pred);
}