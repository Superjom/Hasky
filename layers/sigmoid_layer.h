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
class SigmoidLayer : public Layer<T> {
public:
    typedef typename Layer<T>::vec_t vec_t;

    SigmoidLayer() { }

    virtual void setup(cvshape_t& shapes) {
        const auto& shape = shapes[0];
        CHECK(!this->name().empty()) << "should set layer's name before setup";
        LOG(WARNING) << "construct SigmoidLayer [" << this->name() << "]";
        LOG(WARNING) << "*\tshape:\t" << shape;
        auto& z = this->param().z;
        auto& loss = this->param().loss;
        CHECK_GT(shape.size, 0);
        z.init(shape.size);
        loss.init(shape.size);
    }

    virtual void forward(vec_t& bottom) {
        auto& z = this->param().z;
        CHECK_EQ(bottom.size(), z.size());
        for (int i = 0; i < z.size(); i ++) {
            z[i] = sigmoid<T>(bottom[i]);
        }
    }

    virtual void backward(vec_t& top) {
        auto& z = this->param().z;
        auto& loss = this->param().loss;
        for (int i = 0; i < z.size(); i ++) {
            loss[i] = diff_sigmoid<T>(top[i]);
        }
    }

    virtual void update() { }
};
