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
