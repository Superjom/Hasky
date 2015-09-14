#include <climits>
#include "gtest/gtest.h"
#include "../../layers/neuron_layer.h"

TEST(neuron_layer, init) {
    NeuronLayer<float> _layer;
}
TEST(neuron_layer, setup) {
    NeuronLayer<float> layer;
    layer.gaus_dist().init(0, 0.5);
    shape_t shape(2, 3);
    layer.set_name("ne-layer");
    vector<shape_t> shapes = {shape};
    layer.setup(shapes);
    ASSERT_EQ(layer.param().w.size(), shape.size);
}
TEST(neuron_layer, forward) {
    NeuronLayer<float> layer;
    shape_t shape(2, 3);
    layer.set_name("ne-layer");
    vector<shape_t> shapes = {shape};
    LOG(INFO) << "setup ..";
    layer.gaus_dist().init(0, 0.5);
    layer.setup(shapes);
    LOG(INFO) << "forward ..";
    NeuronLayer<float>::param_t bottom_;
    auto& bottom = bottom_.z;
    bottom.init(3);
    bottom[0] = 1;
    bottom[1] = 2;
    bottom[2] = 3;
   
    for (int i = 0; i < shape.size; i++) {
        LOG(INFO) << "param:\tw" << i << "\t"  << layer.param().w[i];
    }

    layer.forward(bottom_);

    for (int i = 0; i < shape.size; i++) {
        LOG(INFO) << "z:\t" <<i << "\t" << layer.param().z[i];
    }
}
TEST(neuron_layer, backward) {
    NeuronLayer<float> layer;
    shape_t shape(2, 3);
    layer.set_name("ne-layer");
    vector<shape_t> shapes = {shape};
    LOG(INFO) << "setup ..";
    layer.gaus_dist().init(0, 0.5);
    layer.setup(shapes);

    CHECK_EQ(layer.param().loss.size(), 2); 
    NeuronLayer<float>::param_t top;
    auto& top_loss = top.loss;
    top_loss.init(2);
    top_loss[0] = 1; top_loss[1] = 2;
    layer.backward(top);

    for (int i = 0; i < shape.size; i++) {
        LOG(INFO) << i << "th" << "\t" << layer.param().loss[i];
    }
}
