#pragma once
/*
 * neuron_network_layer.h
 *
 *  Created on: Sep 18, 2015
 *      Author: Superjom
 *      Email: yanchunwei@outlook.com
 */
#include "../utils/all.h"
#include "../layer.h"
#include "neuron_layer.h"
#include "map_layer.h"
#include "layer_test.h"


template<typename T>
class NeuronNetworkLayer : public Layer<T> {
public:
    typedef typename Layer<T>::vec_t    vec_t;
    typedef typename Layer<T>::param_t  param_t;

    NeuronNetworkLayer<T>() { 
        this->set_kind(HIDDEN_LAYER);
    }

    void setup(const shape_t& shape_) {
        // init shape
        this->_shape = shape_;
        // init layer
        shape_t shape = this->_shape;
        CHECK(!this->name().empty()) << "should set layer's name before setup";
        LOG(WARNING) << "construct NeuronNetworkLayer [" << this->name() << "]";
        LOG(WARNING) << "*\tshape:\t" << shape;

        neuron_layer.set_name(swift_snails::format_string("%s-%s", this->name().c_str(), "neuron"));
        tanh_layer.set_name(swift_snails::format_string("%s-%s", this->name().c_str(), "tanh"));

        neuron_layer.gaus_dist().init(0, 0.7);
        neuron_layer.setup(shape);
        shape.width = shape.size; 
        tanh_layer.setup(shape);
        // set relationship
        neuron_layer.set_top_layer(&tanh_layer);
        tanh_layer.set_bottom_layer(&neuron_layer);
        // set neural networks param
        this->param().set_z(tanh_layer.param().z_());
        this->param().set_loss(neuron_layer.param().loss_());
    }

    virtual void forward(param_t& bottom) {
        neuron_layer.forward(bottom);    
        tanh_layer.forward(neuron_layer.param());
    }

    virtual void backward(param_t& top, param_t& bottom) {
        tanh_layer.backward(top, neuron_layer.param());
        neuron_layer.backward(tanh_layer.param(), bottom);
    }
    // get the original loss
    T get_weight_loss(int i, int j) {
        return this->param().w()[i][j];
    }

    void update() { }

private:
    NeuronLayer<T> neuron_layer;
    TanhLayer<T> tanh_layer;
};

template<typename TT>
class TestNeuralNetworkLayer : public TestLayer<TT> {
public:
    typedef typename Layer<TT>::vec_t    vec_t;
    typedef typename Layer<TT>::param_t  param_t;

    TestNeuralNetworkLayer<TT>(const string& name, shape_t shape) : TestLayer<TT>(shape.size) 
    {
        layer().set_name(name);
        layer().gaus_dist().init(0, 0.7);
        layer().setup(shape);
    }

    bool loss_grad_check() {
        int input_id = 3;
        float left = this->loss_forward(input_id, 1. - EPISILON);
        float right = this->loss_forward(input_id, 1. + EPISILON);
        float target = this->loss_backward(input_id);
        return grad_check(left, right, target);
    }
    virtual Layer<TT>& layer() {
        return _layer;
    }
    virtual TT get_loss(int input_id) {
        return layer().param().loss()[input_id];
    }
    /*
    virtual void loss_forward(param_t& param) {
        layer().forward(param);
    }
    virtual void loss_backward(param_t& top, param_t& bottom) {
        layer().backward(top, bottom);
    }
    */

protected:
    NeuronNetworkLayer<TT> _layer;
};
