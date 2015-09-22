#pragma once
/*
 * layer_test.h
 *
 *  Created on: Sep 22, 2015
 *      Author: Superjom
 *      Email: yanchunwei@outlook.com
 */
#include "../utils/all.h"
#include "../layer.h"
#include "neuron_layer.h"
#include "map_layer.h"

/*
 * use a simple logistic regression layer to 
 * check grad
 */
template <typename T>
class TestLayer {
public:
    typedef LayerParam<T> param_t;
    typedef Vec<T> vec_t;

    TestLayer(int size) : size(size) {
        // init logistic_layer
        logistic_layer.set_name("logistic layer");
        logistic_layer.gaus_dist().init(0., .7);
        shape_t logistic_shape = { 1, size };
        logistic_layer.setup(logistic_shape);
        // init sigmoid layer
        //shape_t sigmoid_shape = { 1, 1 };
        //sigmoid_layer.set_name("logistic-sigmoid");
        //sigmoid_layer.setup(sigmoid_shape);
    }

    T loss_forward(int input_id, T v) {
        CHECK_LT(input_id, layer().shape().width);
        param_t param;    
        param.z().init(layer().shape().width);
        param.z()[input_id] = v;
        layer().forward(param);
        logistic_layer.forward(layer().param());
        //sigmoid_layer.forward(logistic_layer.param());
        //return sigmoid_layer.param().z()[0];
        return logistic_layer.param().z()[0];
    }
    T loss_backward(int input_id) {
        // init top input
        param_t param;
        param.loss().init(1);
        param.loss()[0] = 1.;
        // init bottom input
        param_t bparam;
        bparam.z().init(layer().shape().width);
        bparam.z()[input_id] = 1.;
        // forward
        layer().forward(bparam);
        logistic_layer.forward(layer().param());
        //sigmoid_layer.forward(logistic_layer.param());
        // backward
        //sigmoid_layer.backward(param, logistic_layer.param());
        //logistic_layer.backward(sigmoid_layer.param(), layer().param());
        logistic_layer.backward(param, layer().param());
        layer().backward(logistic_layer.param(), bparam);
        return get_loss(input_id);
    }

    //virtual void loss_forward(param_t& bottom) = 0;
    //virtual void loss_backward(param_t& top, param_t& bottom) = 0;
    //virtual const shape_t& layer_shape() = 0;
    // layer to test
    virtual Layer<T>& layer() = 0;
    virtual T get_loss(int id) = 0;

protected:
    void __neuron_weight_clear_and_set(param_t &param, int size_id, int width_id, float v) {
        for(int i = 0; i < param.w().size(); i++) {
            for (int j = 0; j < param.w()[0].size(); j++) {
                if (i == size_id && j == width_id) {
                    param.w()[i][j] = v;
                } else {
                    param.w()[i][j] = 0.;
                }
            }
        }
    }



protected:
    NeuronLayer<T> logistic_layer;
    //SigmoidLayer<T> sigmoid_layer;
    int size;
};
