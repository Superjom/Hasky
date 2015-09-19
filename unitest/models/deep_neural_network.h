#include "../../models/deep_neural_network.h"

TEST(DeepNeuralNetworks, setup) {
    DeepNeuralNetworks<float> model;
    vector<int> sizes = {10, 6, 3, 1};
    model.setup(sizes);
}

TEST(DeepNeuralNetworks, learn) {
    DeepNeuralNetworks<float> model;
    vector<int> sizes = {10, 6, 3, 1};
    model.setup(sizes);
    
    DeepNeuralNetworks<float>::vec_t input(10);
    for (int i = 0; i < 10; i++) {
        input[i] = i * 0.1;
    }
    model.learn(input, 1.);
}
