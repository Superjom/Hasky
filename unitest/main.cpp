// utils
#include "utils/vec.h"
//#include "utils/mat.h"
#include "utils/math.h"
#include "utils/random.h"
#include "utils/common.h"
#include "layers/neuron_layer.h"
#include "layers/sigmoid_layer.h"
#include "layers/tanh_layer.h"
#include "layers/loss_layer.h"
#include "layers/data_layer.h"
#include "models/logistic_regression.h"
#include "layers/neural_network_layer.h"
#include "models/deep_neural_network.h"

int main(int argc, char **argv) {  

    testing::InitGoogleTest(&argc, argv);  
    return RUN_ALL_TESTS();  
} 

