#pragma once
/*
 * random.hpp
 *
 *  Created on: Sep 10, 2015
 *      Author: Superjom
 *      Email: yanchunwei@outlook.com
 */

#include <random>


template<typename T>
class GaussianDistrib {
public:
    GaussianDistrib() { }
    GaussianDistrib(T mean, T std) : \
        _mean(mean),
        _std(std)
    { }
    
    T gen() {
        CHECK_GT(_std, 0) << "mean, std should be set";
        static std::mt19937 e2(rd());
        static std::normal_distribution<T> dist(_mean, _std);
        static auto gen_gaussian = std::bind(dist, e2);
        T gaussian = gen_gaussian();
        return gaussian;
    }

    void fill(Vec<T> &vec) {
        CHECK_GT(_std, 0) << "mean, std should be set";
        static std::mt19937 e2(rd());
        static std::normal_distribution<T> dist(_mean, _std);
        static auto gen_gaussian = std::bind(dist, e2);
        for (int i=0; i<vec.size(); i++) {
            vec[i] = gen_gaussian();
        }
    }

    void init(T mean, T std) {
        _mean = mean;
        _std = std;
    }

private:
    std::random_device rd;
    T _mean = 0, _std = 0;
};
