#include "../../layers/loss_layer.h"

TEST(LossLayer, setup) {
    RMSELayer<float> layer; 
    layer.set_name("RMSE layer");
    vector<shape_t> shapes = { shape_t(1, 1)};
    layer.setup(shapes);
}

TEST(LossLayer, forward) {
    RMSELayer<float> layer; 
    layer.set_name("RMSE layer");
    vector<shape_t> shapes = { shape_t(1, 1)};
    layer.setup(shapes);

    layer.param().label[0] = 1.;
    RMSELayer<float>::param_t param;
    param.z.init(1);
    param.z[0] = 0.5;

    layer.forward(param);
    ASSERT_EQ(0.25, layer.param().z[0]);
}

TEST(LossLayer, backward) {
    RMSELayer<float> layer; 
    layer.set_name("RMSE layer");
    vector<shape_t> shapes = { shape_t(1, 1)};
    layer.setup(shapes);

    layer.param().label[0] = 1.;
    RMSELayer<float>::param_t param;
    param.z.init(1);
    param.z[0] = 0.5;
    layer.forward(param);

    layer.backward(param);
    ASSERT_EQ(-1., layer.param().loss[0]);
}
