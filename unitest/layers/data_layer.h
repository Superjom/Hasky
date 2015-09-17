#include "../../layers/data_layer.h"

TEST(DataLayer, setup) {
    DataLayer<float> data_layer;
    data_layer.set_name("data-layer");
    data_layer.setup(10);
    DataLayer<float>::vec_t input(10);
    int index = 5;
    input[index] = .5;
    data_layer.forward(input);
    CHECK_EQ(data_layer.param().z[index], .5);
}
