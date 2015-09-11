/*
 * dataframe.h
 *
 *  Created on: Sep 11, 2015
 *      Author: Superjom
 *      Email: yanchunwei@outlook.com
 */
#pragma once
#include "common.h"
#include "vec.h"

template <typename T>
class DataFrame {
public:
    typedef T value_type;
    typedef Vec<T> vec_t;
    typedef DataFrame<T> self_t;
    typedef vector<T> data_t;

    explicit DataFrame() { }
    explicit DataFrame(int size, int width) {
        set_shape(shape_t(size, width)); 
    }

    void set_shape(const shape_t& shape) {
        CHECK_GT(shape.size, 0);
        CHECK_GT(shape.width, 0);
        // init 
        if (_shape.size != shape.size) {
            _data.resize(shape.size);
            _shape.size = shape.size;
        } 
        if (_shape.width != shape.width) {
            for (int i = 0; i < _shape.size; i++)  {
                _data[i].init(shape.size);
            }
            _shape.width = shape.width;
        }
    }
    const shape_t& shape() const {
        return _shape;
    }
    data_t& data() {
        return _data;
    }
    vec_t& operator[] (int id) {
        CHECK_GE(id, 0);
        CHECK_LT(id, _shape.size);
        return _data[id];
    }
    const vec_t& operator[] (int id) const {
        CHECK_GE(id, 0);
        CHECK_LT(id, _shape.size);
        return _data[id];
    }


private:
    shape_t _shape;
    vector<Vec<value_type> > _data;
};
