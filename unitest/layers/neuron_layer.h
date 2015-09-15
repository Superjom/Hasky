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

    CHECK_EQ(layer.param().loss.size(), 3); 
    NeuronLayer<float>::param_t top;
    auto& top_loss = top.loss;
    top_loss.init(2);
    top_loss[0] = 1; top_loss[1] = 2;
    layer.backward(top);

    for (int i = 0; i < shape.size; i++) {
        LOG(INFO) << i << "th" << "\t" << layer.param().loss[i];
    }
}
TEST(neuron_layer, logistic_loss) {
    NeuronLayer<float> logistic_layer;
    vector<shape_t> logistic_shapes = {shape_t(1, 5)};
    logistic_layer.set_name("logistic layer");
    logistic_layer.gaus_dist().init(0., .7);
    logistic_layer.setup(logistic_shapes);
    NeuronLayer<float>::param_t param;
    param.loss.init(1);
    param.z.init(5);

    for(int index = 0; index < 5; index ++) {
        param.z.clear();
        param.loss.clear();
        param.z[index] = 1. - EPISILON;
        logistic_layer.forward(param);
        float left = logistic_layer.param().z[0];

        param.z[index] = 1. + EPISILON;
        logistic_layer.forward(param);
        float right = logistic_layer.param().z[0];

        param.loss[0] = 1.;
        logistic_layer.backward(param);
        float target = logistic_layer.param().loss[index];

        ASSERT_TRUE(
            grad_check(left, right, target));
    }
}
TEST(neuron_layer, grad_check) {
    // NeuronLayer
    NeuronLayer<float> neuron_layer;
    neuron_layer.set_name("ne-layer");
    vector<shape_t> neuron_shapes = {shape_t(5, 10)};
    LOG(INFO) << "setup ..";
    neuron_layer.gaus_dist().init(0., .7);
    neuron_layer.setup(neuron_shapes);
    // logistic Layer
    NeuronLayer<float> logistic_layer;
    vector<shape_t> logistic_shapes = {shape_t(1, 5)};
    logistic_layer.set_name("logistic layer");
    logistic_layer.gaus_dist().init(0., .7);
    logistic_layer.setup(logistic_shapes);
    // copy weight
    auto w1 = neuron_layer.param().w;
    auto w2 = logistic_layer.param().w;
    // grad check 
    NeuronLayer<float>::param_t param;
    auto& z = param.z;
    auto& loss = param.loss;
    z.init(10);
    loss.init(1);

    for (int index = 0; index < 10; index ++) {
        //neuron_layer.param().w = w1;
        //logistic_layer.param().w = w2;
        // forward
        z.clear();
        loss.clear();
        z[index] = 1. - EPISILON;
        neuron_layer.forward(param);
        logistic_layer.forward(neuron_layer.param());
        float left = logistic_layer.param().z[0];

        z[index] = 1. + EPISILON;
        neuron_layer.forward(param);
        logistic_layer.forward(neuron_layer.param());
        float right = logistic_layer.param().z[0];

        LOG(INFO) << "left:\t" << left << "\tright:\t" << right;
        // backward 
        z.clear();
        loss.clear();
        loss[0] = 1.;
        logistic_layer.backward(param);
        neuron_layer.backward(logistic_layer.param());
        float target = neuron_layer.param().loss[index];
        LOG(INFO) << "target:\t" << target;
        ASSERT_TRUE(
            grad_check(left, right, target)
        );
    }
}
