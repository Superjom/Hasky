#include "../../models/logistic_regression.h"

TEST(LogisticRegression, setup) {
    LogisticRegression<float> model;
    model.setup(10);
}

TEST(LogisticRegression, learn_one_iter) {
    LogisticRegression<float> model;
    model.setup(10);
    LogisticRegression<float>::vec_t input(10);
    float loss, loss2;
    const int iter1 = 10000;
    const int iter2 = 5000;
    for(int i = 0; i < input.size(); i++) {
        input[i] = 0.1 * i;
    }
    ASSERT_GT(iter1, iter2);
    for (int i = 0; i < iter1; i ++) {
        loss = model.learn(input, 1.);
        if (i == iter2) loss2 = loss;
    }
    LOG(INFO) << "learn:\t" << iter1 << " th\tloss:\t" << loss;
    ASSERT_LT(loss, 0.001);
    ASSERT_LT(loss, loss2);
}
