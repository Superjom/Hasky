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


class __LogisticLossTest {
public:
    __LogisticLossTest(int size) : size(size) {
        vector<shape_t> logistic_shapes = {shape_t(1, size)};
        logistic_layer.set_name("logistic layer");
        logistic_layer.gaus_dist().init(0., .7);
        logistic_layer.setup(logistic_shapes);
    }
    float forward(NeuronLayer<float>::param_t &input) {
        logistic_layer.forward(input);
        return logistic_layer.param().z[0];
    }
    float backward(NeuronLayer<float>::param_t &bottom, int id) {
        NeuronLayer<float>::param_t top;
        top.loss.init(1);
        top.loss[0] = 1.;
        logistic_layer.backward(top, bottom);
        //LOG(INFO) << "backward, loss:\t" << logistic_layer.param().loss;
        return logistic_layer.param().loss[id];
    }
    bool test_loss_grad_check(int id) {
        NeuronLayer<float>::param_t bottom;
        bottom.z.init(size);
        bottom.z[id] = 1. - EPISILON;
        float left = forward(bottom);
        bottom.z[id] = 1. + EPISILON;
        float right = forward(bottom);
        bottom.z[id] = 1.;
        float target = backward(bottom, id);
        return grad_check(left, right, target);
    }

private:
    NeuronLayer<float> logistic_layer;
    int size;
};

TEST(NeuronLayer, logistic_loss_grad_check) {
    __LogisticLossTest checker(10);
    ASSERT_TRUE(checker.test_loss_grad_check(5));
}

void __neuron_weight_clear_and_set(NeuronLayer<float>::param_t &param, int size_id, int width_id, float v) {
    for(int i = 0; i < param.w.size(); i++) {
        for (int j = 0; j < param.w[0].size(); j++) {
            if (i == size_id && j == width_id) {
                param.w[i][j] = v;
            } else {
                param.w[i][j] = 0.;
            }
        }
    }
}

class __LogisticWeightGradCheck {
public:
    __LogisticWeightGradCheck(int size) : size(size) {
        vector<shape_t> logistic_shapes = {shape_t(1, size)};
        logistic_layer.set_name("logistic layer");
        logistic_layer.gaus_dist().init(0., .7);
        logistic_layer.setup(logistic_shapes);
        // init input
        input_param.z.init(size);
        for (int i = 0; i < size; i++) {
            input_param.z[i] = .1 * i;
        }
        input_param.loss.init(1);
    }

    float forward(int size_id, int width_id, float v) {
        __neuron_weight_clear_and_set(logistic_layer.param(), size_id, width_id, v);
        logistic_layer.forward(input_param);
        return logistic_layer.param().z[0];
    }

    float backward(int size_id, int width_id, float v) {
        __neuron_weight_clear_and_set(logistic_layer.param(), size_id, width_id, v);
        input_param.loss[0] = 1.;
        logistic_layer.backward(input_param, input_param);
        return logistic_layer.param().w[size_id][width_id];
    }

    void test_weight() {
        int size_id = 0;
        for (int width_id = 0; width_id < size; width_id ++) {
            float left = forward(size_id, width_id, 1. - EPISILON);
            float right = forward(size_id, width_id, 1. + EPISILON);
            float target = (1. - backward(size_id, width_id, 1.) ) / logistic_layer.learning_rate;
            ASSERT_TRUE(grad_check(left, right, target));
        }
    }

private:
    NeuronLayer<float> logistic_layer;
    NeuronLayer<float>::param_t input_param;
    int size;
};


TEST(neuron_layer, logistic_weight_grad_check) {
    __LogisticWeightGradCheck checker(10);
    checker.test_weight();
}

class __TestNeuralLossAndTest {
public:
    typedef NeuronLayer<float>::param_t param_t;

    __TestNeuralLossAndTest(int size, int width) {
        // setup neuron_layer
        neuron_layer.gaus_dist().init(0, 0.5);
        neuron_layer.set_name("neuron-layer");
        neuron_shape.size = size;
        neuron_shape.width = width;
        neuron_layer.set_name("ne-layer");
        vector<shape_t> shapes = {neuron_shape};
        neuron_layer.gaus_dist().init(0., .7);
        neuron_layer.setup(shapes);
        // setup logistic layer
        logistic_shape.size = 1;
        logistic_shape.width = size;
        vector<shape_t> lshapes = {logistic_shape};
        logistic_layer.set_name("logistic-layer");
        logistic_layer.gaus_dist().init(0., .7);
        logistic_layer.setup(lshapes);
    }

    float loss_forward(int input_id, float v) {
        param_t param;    
        param.z.init(neuron_shape.width);
        param.z[input_id] = v;
        neuron_layer.forward(param);
        logistic_layer.forward(neuron_layer.param());
        return logistic_layer.param().z[0];
    }

    float weight_forward(int size_id, int width_id, float v) {
        param_t param;    
        param.z.init(neuron_shape.width);
        for (int i = 0; i < neuron_shape.width; i++) {
            param.z[i] = i * 0.1;
        }
        __neuron_weight_clear_and_set(neuron_layer.param(), size_id, width_id, v);
        neuron_layer.forward(param);
        logistic_layer.forward(neuron_layer.param());
        return logistic_layer.param().z[0];
    }

    float loss_backward(int input_id) {
        param_t param;
        param.loss.init(1);
        param.loss[0] = 1.;

        param_t bparam;
        bparam.z.init(neuron_shape.width);
        bparam.z[input_id] = 1.;

        neuron_layer.forward(bparam);
        logistic_layer.forward(neuron_layer.param());

        logistic_layer.backward(param, neuron_layer.param());

        neuron_layer.backward(logistic_layer.param(), bparam);
        return neuron_layer.param().loss[input_id];
        //return (1. - neuron_layer.param().loss[input_id]) / neuron_layer.learning_rate;
    }

    float weight_backward(int size_id, int width_id, float v) {
        __neuron_weight_clear_and_set(neuron_layer.param(), size_id, width_id, v);
        // init top
        param_t param;
        param.loss.init(1);
        param.loss[0] = 1.;

        param_t bparam;
        bparam.z.init(neuron_shape.width);
        for (int i = 0; i < neuron_shape.width; i++) {
            bparam.z[i] = i * 0.1;
        }
        neuron_layer.forward(bparam);

        logistic_layer.backward(param, neuron_layer.param());
        neuron_layer.backward(logistic_layer.param(), bparam);
        return (1. - neuron_layer.param().w[size_id][width_id]) / neuron_layer.learning_rate;
    }

    void test_loss_grad_check() {
        for( int input_id = 0; input_id < neuron_shape.width; input_id++) {
            float left = loss_forward(input_id, 1. - EPISILON);
            float right = loss_forward(input_id, 1. + EPISILON);
            float target = loss_backward(input_id);
            ASSERT_TRUE(
                grad_check(left, right, target)
            );
        }
    }

    void test_weight_grad_check() {
        for(int size_id = 0; size_id < neuron_shape.size; size_id++) {
            for(int width_id = 0; width_id < neuron_shape.width; width_id++) {
                float left = weight_forward(size_id, width_id, 1. - EPISILON);
                float right = weight_forward(size_id, width_id, 1. + EPISILON);
                float target = weight_backward(size_id, width_id, 1.);
                ASSERT_TRUE(
                    grad_check(left, right, target)
                );
            }
        }
    }

private:
    NeuronLayer<float> neuron_layer;
    NeuronLayer<float> logistic_layer;
    shape_t neuron_shape;
    shape_t logistic_shape;
};

TEST(NeuronLayer, loss_grad_check) {
    __TestNeuralLossAndTest checker(10, 20);
    checker.test_loss_grad_check();
}

TEST(NeuronLayer, weight_grad_check) {
    __TestNeuralLossAndTest checker(10, 20);
    checker.test_weight_grad_check();
}
