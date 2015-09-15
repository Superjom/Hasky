#pragma once
/*
 * loss_layer.h
 *
 *  Created on: Sep 15, 2015
 *      Author: Superjom
 *      Email: yanchunwei@outlook.com
 */
#include "../utils/all.h"
#include "../layer.h"

template<typename T>
class LossLayer : public Layer<T>{
public:
    typedef typename Layer<T>::vec_t    vec_t;
    typedef typename Layer<T>::param_t  param_t;

    LossLayer() { }
    /*
     * set loss's size
     *
     * @param shapes
     *  shapes[0].size: loss.size, label.size
     */
    virtual void setup(cvshape_t& shapes) {
        CHECK(!this->name().empty()) << "should set layer's name before setup";
        LOG(WARNING) << "construct " << " [" << this->name() << "]";
        const auto& shape = shapes[0];
        LOG(WARNING) << "*\tsize:\t" << shape.size;
        this->param().z.init(shape.size);
        this->param().loss.init(shape.size);
        this->param().label.init(shape.size);
    }
    // forward
    // backward
};


template<typename T>
class RMSELayer : public LossLayer<T> {
public:
    typedef typename LossLayer<T>::vec_t    vec_t;
    typedef typename LossLayer<T>::param_t  param_t;
    
    virtual void forward(param_t& bottom_) {
        auto& bottom_z = bottom_.z;
        auto& label = this->param().label;
        auto& z = this->param().z;
        CHECK_EQ(this->param().loss.size(), 1) << "RMSE loss should be a float";
        CHECK_EQ(this->param().label.size(), 1);
        z[0] = pow(bottom_z[0] - label[0], 2);
    }

    virtual void backward(param_t& top_) {
        /*
        T z = this->param().z[0];
        auto& loss = this->param().loss;
        loss[0] =  2. * sqrt(z);
        */
    }

    void backward() {
        T z = this->param().z[0];
        auto& label = this->param().label[0];
        auto& loss = this->param().loss;
        loss[0] =  2. * sqrt(z) * \
                ((z - label) > 0. ? 1. : -1.);
    }

    virtual void update() { }
};
