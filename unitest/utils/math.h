#include <climits>
#include "gtest/gtest.h"
#include "../../utils/all.h"

bool sigmoid_C(float v) {
    float left = sigmoid<float>(v - EPISILON);
    float right = sigmoid<float>(v + EPISILON);
    float target = diff_sigmoid<float>(v); 
    return grad_check<float>(left, right, target);
}
TEST(math, sigmoid_grad_check) {
    GaussianDistrib<float> gau(0., 10.);
    for (int i = 0; i < 10; i++) {
        float v = gau.gen();
        ASSERT_TRUE(sigmoid_C(v));
    }
}

bool tanh_C(float v) {
    float left = tanh<float>(v - EPISILON);
    float right = tanh<float>(v + EPISILON);
    float tanh_v = tanh<float>(v);
    float target = diff_tanh<float>(tanh_v);
    return grad_check<float>(left, right, target);
}
TEST(math, tanh_grad_check) {
    GaussianDistrib<float> gau(0., 10.);
    for (int i = 0; i < 10; i++) {
        float v = gau.gen();
        ASSERT_TRUE(tanh_C(v));
    }
}
