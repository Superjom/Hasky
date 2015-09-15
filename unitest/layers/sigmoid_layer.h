#include <climits>
#include "gtest/gtest.h"
//#include "../../layers/sigmoid_layer.h"
//#include "../../layers/map_layer2.h"
#include "../../layers/map_layer.h"

TEST(sigmoid_layer, setup) {
    SigmoidLayer<float> layer;
    shape_t shape(2, 3);
    layer.set_name("sigmoid-layer");
    vector<shape_t> shapes = {shape};
    layer.setup(shapes);
    ASSERT_TRUE(layer.param().w.empty());
    ASSERT_EQ(layer.param().z.size(), shape.size);
    ASSERT_EQ(layer.param().loss.size(), shape.size);
}

TEST(sigmoid_layer, forward) {
    SigmoidLayer<float> layer;
    shape_t shape(5, 5);
    layer.set_name("sigmoid-layer");
    vector<shape_t> shapes = {shape};
    layer.setup(shapes);

    SigmoidLayer<float>::param_t param;
    auto& x = param.z;
    auto& loss = param.loss;
    x.init(shape.size);
    loss.init(shape.size);
    for (int i = 0; i < shape.size; i++) {
        x[i] = i+1;
        loss[i] = i + 1;
    }

    layer.forward(param);
    LOG(INFO) << "forward z:\t" << layer.param().z;

    layer.backward(param);
    LOG(INFO) << "backward loss:\t" << layer.param().loss;
}

TEST(sigmoid_layer, grad_check) {
    SigmoidLayer<float> layer;
    shape_t shape(10, 10);
    layer.set_name("sigmoid-layer");
    vector<shape_t> shapes = {shape};
    layer.setup(shapes);

    SigmoidLayer<float>::param_t param;
    auto& bottom_z = param.z;
    auto& loss = param.loss;
    bottom_z.init(10);
    loss.init(10);

    for (int index = 0; index < 10; index++) {
        bottom_z.clear(); 
        loss.clear();
        bottom_z[index] = 1. + EPISILON;
        layer.forward(param);
        float right = layer.param().z[index];

        bottom_z[index] = 1. - EPISILON;
        layer.forward(param);
        float left = layer.param().z[index];
       
        loss[index] = 1.;
        layer.backward(param);
        float target = layer.param().loss[index];

        ASSERT_TRUE(
            grad_check(left, right, target));
    }
}
