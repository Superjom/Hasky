/*
 * vec.hpp
 *
 *  Created on: Sep 11, 2015
 *      Author: Superjom
 *      Email: yanchunwei@outlook.com
 */
#include "utils/common.hpp"
#include "utils/dataframe.h"

template<typename T>
struct LayerParam {
    typedef T value_type;

    DataFrame<value_type> data;
    //DataFrame<value_type> bottom;
    DataFrame<value_type> loss;
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
    typedef Layer<T> layer_t;
    typedef DataFrame<T> df_t;

    const string& get_name() const { return _name; }
    void set_name(const string& name) { 
        CHECK( ! name.empty());
        _name = name; 
    }

    virtual void setup(df_t& shapes) = 0;
    virtual void forward(df_t& bottom, df_t& top) = 0;
    virtual void backward(df_t& top, df_t& bottom) = 0;

    param_t& param() { return _param; }
    const param_t& param() const { return _param; }
    
    void set_top_layer(layer_t* top) {
        _top_layer = top;
    }
    void set_bottom_layer(layer_t* bottom) {
        _bottom_layer = bottom;
    }
    layer_t* top_layer() { return _top_layer; }
    layer_t* bottom_layer() { return _bottom_layer; }

private:
    string _name;
    param_t _param;
    layer_t* _top_layer = nullptr; 
    layer_t* _bottom_layer = nullptr;
};
