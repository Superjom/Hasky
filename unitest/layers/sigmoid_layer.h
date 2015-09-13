#include <climits>
#include "gtest/gtest.h"
#include "../../layers/sigmoid_layer.h"

TEST(sigmoid_layer, init) {
    SigmoidLayer<float> layer;
    shape_t shape(2, 3);
    layer.set_name("sigmoid-layer");
    vector<shape_t> shapes = {shape};
    layer.setup(shapes);
    ASSERT_TRUE(layer.param().w.empty());
    ASSERT_EQ(layer.param().z.size(), shape.size);
    ASSERT_EQ(layer.param().loss.size(), shape.size);
}
