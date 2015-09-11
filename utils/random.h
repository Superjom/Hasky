#pragma once
/*
 * random.hpp
 *
 *  Created on: Sep 10, 2015
 *      Author: Superjom
 *      Email: yanchunwei@outlook.com
 */

#include <random>



class GaussianDistrib {
public:
    GaussianDistrib(float mean, float std) : \
        _mean(mean),
        _std(std)
    { }
    
    float gen() {
        std::mt19937 e2(rd());
        std::normal_distribution<float> dist(_mean, _std);
        auto gen_gaussian = std::bind(dist, e2);
        float gaussian = gen_gaussian();
        return gaussian;
    }

private:
    std::random_device rd;
    float _mean, _std;
};
