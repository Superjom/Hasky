#pragma once
#include "../layers/all.h"

template<typename T>
class LogisticRegression {
public:
    typedef typename Layer<T>::param_t param_t;
    typedef typename Layer<T>::vec_t vec_t;
    /*
     * @size: length of hidden vector
     */
    void setup(int size) {
        vector<shape_t> shapes = { shape_t(1, size) };
        neuron_layer.set_name("neural-layer");
        neuron_layer.gaus_dist().init(0, 0.5);
        neuron_layer.setup(shapes);
        shapes[0].size = 1;
        shapes[0].width = 1;

        sigmoid_layer.set_name("sigmoid-layer");
        sigmoid_layer.gaus_dist().init(0, 0.5);
        sigmoid_layer.setup(shapes);

        loss_layer.set_name("rmse-layer");
        loss_layer.gaus_dist().init(0, 0.5);
        loss_layer.setup(shapes);

        data_layer.set_name("data-layer");
        data_layer.setup(size);
    }

    float learn(vec_t& vec, T label) {
        loss_layer.param().label()[0] = label;
        data_layer.forward(vec);
        neuron_layer.forward(data_layer.param());
        sigmoid_layer.forward(neuron_layer.param());
        loss_layer.forward(sigmoid_layer.param());
        float loss = loss_layer.param().z()[0];
        // backward
        loss_layer.backward(sigmoid_layer.param());
        sigmoid_layer.backward(loss_layer.param(), neuron_layer.param());
        neuron_layer.backward(sigmoid_layer.param(), data_layer.param());
        return loss;
    }

    T predict(vec_t& vec) {
        data_layer.forward(vec);
        neuron_layer.forward(data_layer.param());
        sigmoid_layer.forward(neuron_layer.param());
        loss_layer.forward(sigmoid_layer.param());
        return sigmoid_layer.param().z()[0];
    }

private:
    NeuronLayer<T> neuron_layer;
    SigmoidLayer<T> sigmoid_layer;
    RMSELayer<T> rmse_loss_layer;
    DataLayer<T> data_layer;
    RMSELayer<T> loss_layer;
};
