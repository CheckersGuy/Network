//
// Created by root on 14.03.20.
//

#include "Network.h"
#include "Utils.h"

std::mt19937 generator(12314124ull);

void Layer::forward_pass(float *input) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, num_rows, num_cols, 1.0f, weights.get(), num_cols, input, 1, 0.0f,
                pre_activ.get(), 1);

    cblas_saxpy(num_rows, 1.0f, biases.get(), 1, pre_activ.get(), 1);

    //computing the post-activation
    activation();
}

void Network::forward_pass(float *input) {
    Layer &first_layer = layers.front();
    first_layer.forward_pass(input);

    for (auto i = 1; i < layers.size(); ++i) {
        Layer &previous = layers[i - 1];
        layers[i].forward_pass(previous.post_activ.get());
    }
}


Layer &Network::get_output_layer() {
    return layers.back();
}

const Layer &Network::get_output_layer() const {
    return layers.back();
}