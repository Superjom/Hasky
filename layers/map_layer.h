#pragma once
/*
 * map_layer.h
 *
 *  Created on: Sep 13, 2015
 *      Author: Superjom
 *      Email: yanchunwei@outlook.com
 */
#include "../utils/all.h"
#include "../layer.h"

#define CREATE_MAP_LAYER(FOR_MAP,BAC_MAP,NAME)              \
template<typename T>                                        \
class NAME ## Layer : public Layer<T> {                     \
public:                                                     \
    typedef typename Layer<T>::vec_t vec_t;                 \
    NAME ## Layer() { }                                          \
                                                            \
    virtual void setup(cvshape_t& shapes) {                 \
        const auto& shape = shapes[0];                      \
        this->set_name(#NAME);                               \
        CHECK(!this->name().empty()) << "should set layer's name before setup"; \
        LOG(WARNING) << "construct " << #NAME << "Layer [" << this->name() << "]"; \
        LOG(WARNING) << "*\tshape:\t" << shape;             \
        auto& z = this->param().z;                          \
        auto& loss = this->param().loss;                    \
        CHECK_GT(shape.size, 0);                            \
        z.init(shape.size);                                 \
        loss.init(shape.size);                              \
    }                                                       \
                                                            \
    virtual void forward(vec_t& bottom) {                   \
        auto& z = this->param().z;                          \
        CHECK_EQ(bottom.size(), z.size());                  \
        for (int i = 0; i < z.size(); i ++) {               \
            z[i] = FOR_MAP<T>(bottom[i]);                   \
        }                                                   \
    }                                                       \
                                                            \
    virtual void backward(vec_t& top) {                     \
        auto& z = this->param().z;                          \
        auto& loss = this->param().loss;                    \
        for (int i = 0; i < z.size(); i ++) {               \
            loss[i] = BAC_MAP<T>(top[i]);                   \
        }                                                   \
    }                                                       \
                                                            \
    virtual void update() { }                               \
};

CREATE_MAP_LAYER(sigmoid, diff_sigmoid, Sigmoid)
