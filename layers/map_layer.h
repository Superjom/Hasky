#pragma once
/*
 * sigmoid_layer.h
 *
 *  Created on: Sep 13, 2015
 *      Author: Superjom
 *      Email: yanchunwei@outlook.com
 */
#include "../utils/all.h"
#include "../layer.h"

template<typename T>
class MapLayer : public Layer<T> {
public:
    typedef typename Layer<T>::vec_t    vec_t;
    typedef typename Layer<T>::param_t  param_t;


    MapLayer() { }

    virtual void setup(cvshape_t& shapes) {
        const auto& shape = shapes[0];
        //this->set_name(NAME);
        CHECK(!this->name().empty()) << "should set layer's name before setup";
        LOG(WARNING) << "construct " << " [" << this->name() << "]";
        LOG(WARNING) << "*\tshape:\t" << shape;
        auto& z = this->param().z;
        auto& loss = this->param().loss;
        CHECK_GT(shape.size, 0);
        z.init(shape.size);
        loss.init(shape.size);
    }

    virtual void forward(param_t& bottom_) {
        auto& bottom = bottom_.z;
        auto& z = this->param().z;
        CHECK_EQ(bottom.size(), z.size());
        for (int i = 0; i < z.size(); i ++) {
            // TODO add map
            z[i] = (bottom[i]);
        }
    }

    virtual void backward(param_t& top_) {
        auto& top = top_.loss;
        auto& z = this->param().z;
        auto& loss = this->param().loss;
        for (int i = 0; i < z.size(); i ++) {
            // TODO to add map
            loss[i] = (top[i]);
        }
    }

    virtual void update() { }
};   // end MapLayer

//const char* sigmoid_layer_name = "SigmoidLayer";

template<typename T>
class SigmoidLayer : public MapLayer<T>
{
public:
    typedef typename MapLayer<T>::param_t param_t;

    virtual void forward(param_t& bottom_) {
        auto& bottom = bottom_.z;
        auto& top = this->param().z;
        CHECK_EQ(bottom.size(), top.size());
        for (int i = 0; i < top.size(); i ++) {
            top[i] = sigmoid<T>(bottom[i]);
        }
    }

    virtual void backward(param_t& top_) {
        auto& top = top_.loss;
        auto& bottom = this->param().loss;
        CHECK_EQ(top.size(), bottom.size());
        for (int i = 0; i < top.size(); i ++) {
            bottom[i] = diff_sigmoid<T>(top[i]);
        }
    }
};  // end SigmoidLayer
