#pragma once 

namespace FAST_INFERENCE {
    #define N_CLASSES 10
    #define N_FEATURES 3072

    void predict_model(double const * const x, double * pred);
}