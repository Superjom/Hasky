#pragma once
/*
 * layer.h
 *
 *  Created on: Sep 11, 2015
 *      Author: Superjom
 *      Email: yanchunwei@outlook.com
 */
#include "utils/all.h"

template<typename T>
struct LayerParam {
    typedef T value_type;
    typedef DataFrame<T> df_t;
    typedef Vec<T> vec_t;

    vec_t z;
    vec_t loss;
    vec_t label;    // used by loss layer
    df_t w; 
    //DataFrame<value_type> bottom;
};
/*
 * a layer should implement:
 *      forward
 *      backward
 *      setup
 */
template<typename T>
class Layer {
public:
    typedef T value_type;
    typedef LayerParam<value_type> param_t;
    typedef Vec<T> vec_t;
    typedef Layer<T> layer_t;
    typedef DataFrame<T> df_t;

    const string& name() const { return _name; }
    void set_name(const string& name) { 
        CHECK( ! name.empty());
        _name = name; 
    }
    virtual void setup(const shape_t& shape) { }
    virtual void setup(cvshape_t& shapes) { }
    virtual void forward(param_t& bottom) { }
    /*
     * top: top gradient
     * bottom: bottom gradient
     *
     * top gradient * cur_grad -> bottom gradient
     */
    virtual void backward(param_t& top) { }
    virtual void backward(param_t& top, param_t& bottom) { }
    /*
     * add current layer's loss to it's parameter
     */
    virtual void update() = 0;

    virtual param_t& param() { return _param; }
    virtual const param_t& param() const { return _param; }
    
    void set_top_layer(layer_t* top) {
        this->_top_layer = top;
    }
    void set_bottom_layer(layer_t* bottom) {
        this->_bottom_layer = bottom;
    }
    layer_t* top_layer() { 
        return this->_top_layer; 
    }
    layer_t* bottom_layer() { 
        return this->_bottom_layer; 
    }
    virtual ~Layer() { }

    static const float learning_rate;

    GaussianDistrib<T>& gaus_dist() {
        return _gaus_dist;
    }

    layer_kind_t kind() {
        return _kind;
    }

    void set_kind(layer_kind_t kind) {
        _kind = kind;
    }

private:
    string _name;
    param_t _param;
    layer_t* _top_layer = nullptr; 
    layer_t* _bottom_layer = nullptr;
    GaussianDistrib<T> _gaus_dist;
    layer_kind_t _kind;
};
template<typename T>
const float Layer<T>::learning_rate = 0.02;
