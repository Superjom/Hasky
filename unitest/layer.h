#include "../layer.h"

template<class T>
class TestLayer : public Layer<T> {
public:
    void setup(vector<shape_t>& shapes) {
        LOG(INFO) << "shapes:\t";
        for(auto& shape : shapes) {
            LOG(INFO) << shape;
        }
    }
    void forward(df_t& bottom, df_t& top) {
    }
    void backward(df_t& top, df_t& bottom) {
    }
};
