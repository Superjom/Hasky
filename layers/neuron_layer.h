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
    typedef typename Layer<T>::vec_t    vec_t;
    typedef typename Layer<T>::param_t  param_t;

    NeuronLayer<T>() { 
        this->set_kind(HIDDEN_LAYER);
    }

    void setup(const shape_t &shape) {
        CHECK(!this->name().empty()) << "should set layer's name before setup";
        LOG(WARNING) << "construct NeuronLayer [" << this->name() << "]";
        LOG(WARNING) << "*\tshape:\t" << shape;
        //const auto& shape = shapes[0];
        // construct parameter
        //CHECK(this->param().w.empty());
        this->param().w().set_shape(shape);
        this->param().z().init(shape.size);
        this->param().loss().init(shape.width);
        // random init
        for (auto& vec : this->param().w().data()) 
            this->gaus_dist().fill(vec);
    }
    
    virtual void setup(cvshape_t& shapes) {
        setup(shapes[0]);
    }
    /*
     * from bottom's z to top's z
     */
    virtual void forward(param_t& bottom_) {
        auto& bottom = bottom_.z();
        auto& top = this->param().z();
        CHECK(! bottom.empty()) << this->name();
        CHECK(! top.empty()) << this->name();
        // shape check
        CHECK_EQ(top.size(), this->param().w().size());
        CHECK_EQ(bottom.size(), this->param().w()[0].size());
        // compute
        for(int i = 0; i < top.size(); i++) {
            top[i] = bottom.dot( this->param().w()[i]);
        }
    }
    /*
     * top: top loss
     * bottom: bottom input
     */
    virtual void backward(param_t& top, param_t& bottom) {
        // update x
        auto& param = this->param();
        auto& w = param.w();
        CHECK_EQ(top.loss().size(), w.size());
        CHECK_EQ(bottom.z().size(), w[0].size());

        param.loss().clear();
        // TODO assert loss == 0 ? 
        for (int i = 0; i < param.loss().size(); i++) {
            for (int j = 0; j < top.loss().size(); j++) {
                // TODO assign loss or accumulate loss ? 
                param.loss()[i] += w[j][i] * top.loss()[j];
                //LOG(INFO) << "loss" << i << "\t" << param.loss[i] << "\tw" << j << i << "\t" << w[j][i] << "\ttop.loss" << j << "\t" << top.loss[j];
            }
        }
        // update weight
        // TODO update weight !
        for (int i = 0; i < bottom.z().size(); i++) {
            for (int j = 0; j < top.loss().size(); j++) {
                w[j][i] -= this->learning_rate * top.loss()[j] * bottom.z()[i];
            }
        }
    }

    virtual void update() {
    }

    virtual ~NeuronLayer() { }

};  // end class NeuronLayer

