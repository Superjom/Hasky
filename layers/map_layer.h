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

template<typename T>
class MapLayer : public Layer<T> {
public:
    typedef typename Layer<T>::vec_t    vec_t;
    typedef typename Layer<T>::param_t  param_t;


    MapLayer() { 
        this->set_kind(HIDDEN_LAYER);
    }

    virtual void setup(const shape_t& shape) {
        //this->set_name(NAME);
        CHECK(!this->name().empty()) << "should set layer's name before setup";
        LOG(WARNING) << "construct " << " [" << this->name() << "]";
        LOG(WARNING) << "*\tshape:\t" << shape;
        auto& z = this->param().z();
        auto& loss = this->param().loss();
        CHECK_GT(shape.size, 0);
        z.init(shape.size);
        loss.init(shape.size);
    }

    virtual void setup(cvshape_t& shapes) {
        const auto& shape = shapes[0];
        setup(shape);
    }

    virtual void forward(param_t& bottom_) {
        auto& bottom = bottom_.z();
        auto& z = this->param().z();
        CHECK_EQ(bottom.size(), z.size());
        for (int i = 0; i < z.size(); i ++) {
            // TODO add map
            z[i] = (bottom[i]);
        }
    }

    virtual void backward(param_t& top, param_t& bottom) = 0;

    virtual void update() { }
};   // end MapLayer

//const char* sigmoid_layer_name = "SigmoidLayer";

template<typename T>
class SigmoidLayer : public MapLayer<T>
{
public:
    typedef typename MapLayer<T>::param_t param_t;

    SigmoidLayer() {
        REGISTER_LAYER("sigmoid layer", SigmoidLayer<T>)
    }

    virtual void forward(param_t& bottom_) {
        auto& bottom = bottom_.z();
        auto& top = this->param().z();
        CHECK_EQ(bottom.size(), top.size());
        for (int i = 0; i < top.size(); i ++) {
            top[i] = sigmoid<T>(bottom[i]);
        }
    }

    virtual void backward(param_t& top_, param_t& bottom) {
        auto& top = top_.loss();
        auto& z = bottom.z();
        auto& loss = this->param().loss();
        CHECK_EQ(top.size(), loss.size());
        for (int i = 0; i < top.size(); i ++) {
            loss[i] = diff_sigmoid<T>(z[i]) * top[i];
        }
    }
    LAYER_INIT_INSIDE_CLASS
};  // end SigmoidLayer
LAYER_INIT_OUTSIDE_CLASS(SigmoidLayer)

template<typename T>
class TanhLayer : public MapLayer<T>
{
public:
    typedef typename MapLayer<T>::param_t param_t;

    TanhLayer() {
        REGISTER_LAYER("tanh layer", TanhLayer)
    }

    virtual void forward(param_t& bottom_) {
        auto& bottom = bottom_.z();
        auto& top = this->param().z();
        CHECK_EQ(bottom.size(), top.size());
        for (int i = 0; i < top.size(); i ++) {
            top[i] = tanh<T>(bottom[i]);
        }
    }

    virtual void backward(param_t& top, param_t& bottom) {
        auto& top_loss = top.loss();
        auto& loss = this->param().loss();
        auto& z = this->param().z();
        CHECK_EQ(top_loss.size(), loss.size());
        for (int i = 0; i < loss.size(); i ++) {
            loss[i] = diff_tanh<T>(z[i]) * top_loss[i];
        }
    }

private:
    LAYER_INIT_INSIDE_CLASS
};
LAYER_INIT_OUTSIDE_CLASS(TanhLayer)
