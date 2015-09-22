#include "../../layers/neural_network_layer.h"

TEST(NeuronNetworkLayer, loss_grad_check) {
    TestNeuralNetworkLayer<float> layer("neural-network-layer-test", shape_t(10, 5));
    layer.loss_grad_check();
}
