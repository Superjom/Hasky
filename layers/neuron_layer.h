#pragma once
/*
 * neuron_layer.h
 *
 *  Created on: Sep 11, 2015
 *      Author: Superjom
 *      Email: yanchunwei@outlook.com
 */
#include "../utils/all.h"
#include "../layer.h"

template<typename T>
class NeuronLayer : public Layer<T> {
public:
    typedef typename Layer<T>::vec_t vec_t;

    NeuronLayer<T>() { }

    virtual void setup(cvshape_t& shapes) {
        CHECK(!this->name().empty()) << "should set layer's name before setup";
        LOG(WARNING) << "construct NeuronLayer [" << this->name() << "]";
        LOG(WARNING) << "*\tshape:\t" << shapes[0];
        const auto& shape = shapes[0];
        // construct parameter
        CHECK(this->param().w.empty());
        this->param().w.set_shape(shape);
        this->param().z.init(shape.size);
        this->param().loss.init(shape.size);
        // random init
        for (auto& vec : this->param().w.data()) 
            this->gaus_dist().fill(vec);
    }
    /*
     * from top's loss to bottom's loss
     */
    virtual void forward(vec_t& bottom) {
        auto& top = this->param().z;
        CHECK(! bottom.empty());
        CHECK(! top.empty());
        // shape check
        CHECK_EQ(top.size(), this->param().w.size());
        CHECK_EQ(bottom.size(), this->param().w[0].size());
        // compute
        for(int i = 0; i < top.size(); i++) {
            LOG(INFO) << i << "th";
            top[i] = bottom.dot( this->param().w[i]);
        }
    }
    /*
     * top: top loss
     * bottom: bottom input
     */
    virtual void backward(vec_t& top) {
        // update x
        auto& bottom = this->param().loss;
        T* loss = &this->param().loss[0];
        // TODO assert loss == 0 ? 
        for (int i = 0; i < this->param().loss.size(); i++) {
            for (int j = 0; j < top.size(); j++) {
                loss[i] += this->param().w[j][i] * top[i];
            }
        }
        // update weight
        for (int i = 0; i < this->param().loss.size(); i++) {
            for (int j = 0; j < top.size(); j++) {
                this->param().w[j][i] -= this->learning_rate * bottom[i];
            }
        }
    }

    virtual void update() {
    }

    virtual ~NeuronLayer() { }

};  // end class NeuronLayer
