#pragma once
/*
 * data_layer.h
 *
 *  Created on: Sep 15, 2015
 *      Author: Superjom
 *      Email: yanchunwei@outlook.com
 */
#include "../utils/all.h"
#include "../layer.h"

template<typename T>
class DataLayer : public Layer<T> {
public:
    typedef typename Layer<T>::vec_t    vec_t;
    typedef typename Layer<T>::param_t  param_t;

    DataLayer() { 
        this->set_kind(INPUT_LAYER);
    }

    void setup(int size) {
        CHECK(!this->name().empty()) << "should set layer's name before setup";
        LOG(WARNING) << "construct NeuronLayer [" << this->name() << "]";
        LOG(WARNING) << "*\tshape:\t" << size;
        auto& param = this->param();
        param.z.init(size);
    }
    void forward(const vec_t& input) {
        auto& param = this->param();
        CHECK_EQ(input.size(), param.z.size());
        param.z = input;
    }
    virtual void setup(cvshape_t& shapes) {
    }
    virtual void forward(param_t& param) {
        CHECK(false) << "forward() is deleted";
    }
    virtual void backward(param_t& param) {
        CHECK(false) << "backwrard() is deleted";
    }
    virtual void update() { }
};
