#include "../../models/dnn_factory.h"

TEST(DnnFactory, init) {
    DnnHiddenFactory<float> fac;
}

TEST(DnnFactory, setup) {
    DnnHiddenFactory<float> fac;

    vector<LayerFactItem> layers = {
        LayerFactItem("neuron layer", 10),
        LayerFactItem("neuron layer", 7),
        LayerFactItem("neuron layer", 3)
    };

    fac.set_name("neuron factor");
    fac.setup(shape_t(20, 1));
    fac.setup(layers);
}
