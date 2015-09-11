#pragma once
/*
 * common.h
 *
 *  Created on: Sep 10, 2015
 *      Author: Superjom
 *      Email: yanchunwei@outlook.com
 */
#include <iostream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <random>
using namespace std;
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "gflags/gflags.h"

typedef unsigned int uint;

struct shape_t {
    uint size = -1;
    uint width = -1;

    shape_t() { }
    shape_t(int size, int width) {
        CHECK_GT(size, 0);
        CHECK_GT(width, 0);
        this->size = size;
        this->width = width;
    }
};
