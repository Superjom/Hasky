#include <climits>
#include "gtest/gtest.h"
#include "../../utils/all.h"

TEST(shape, init) {
    shape_t shape(2, 3);
    LOG(INFO) << "shape:\t" << shape;
}
