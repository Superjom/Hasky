#include "../../utils/vec.hpp"
#include <climits>
#include "gtest/gtest.h"

TEST(vec, init) {
    Vec<float> vec1(5);
    for(int i = 0; i < 5; i++) vec1[i] = i;
    Vec<float> vec2(vec1);

    for(int i = 0; i < 5; i++) {
        ASSERT_EQ(vec2[i], i);
    }
}
