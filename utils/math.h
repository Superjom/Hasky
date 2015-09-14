template<typename T>
inline T sigmoid(T v) {
    return 1. / (1. + exp(- v));
}

template<typename T>
inline T diff_sigmoid(T x) {
    float exp_x = exp(x);
    return exp_x / pow((1 + exp_x), 2);
}

template<typename T>
inline T tanh(T x) {
    float ex = exp(x);
    float e_x = 1. / ex;
    return (ex - e_x) / (ex + e_x);
}

template<typename T>
inline T diff_tanh(T x) {
}

