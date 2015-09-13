#include <climits>
#include "gtest/gtest.h"
#include "../../utils/mat.h"

TEST(mat, init) {
    Mat<float> mat;
}
/*
TEST(mat, shape) {
    shape_t shape1(3, 2);
    Mat<float> mat(shape1);
    const auto& shape = mat.shape();
    ASSERT_EQ(shape.size, shape.size);
    ASSERT_EQ(shape.width, shape.width);
}
TEST(mat, assign) {
    shape_t shape(3, 2);
    Mat<float> mat(shape);
    for (int i = 0; i < shape.size; i++) {
        for (int j = 0; j < shape.width; j++) {
            ASSERT_EQ(mat[i][j], 0);
        }
    }
    // assign
    mat[0][1] = 1;

    ASSERT_EQ(mat[0][1], 1);
}
*/
