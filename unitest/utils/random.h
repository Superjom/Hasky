#include <climits>
#include "gtest/gtest.h"
#include "../../utils/random.h"
#include "../../utils/vec.h"

TEST(random, gaussian) {
    GaussianDistrib<float> gau(0.0, 1.0);
    for(int i = 0; i < 10; i++) {
        LOG(INFO) << i << "th " << gau.gen();
    }
}
