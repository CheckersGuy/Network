//
// Created by root on 14.03.20.
//

#include "Network.h"
#include "Utils.h"

std::mt19937 generator(5314124ull);

void Layer::forward_pass(float *input) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, num_rows, num_cols, 1.0f, weights.get(), num_cols, input, 1, 0.0f,
                pre_activ.get(), 1);
    std::transform(pre_activ.get(), pre_activ.get() + num_rows, biases.get(), pre_activ.get(), std::plus<float>{});
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

void Network::init_weights() {

    std::uniform_real_distribution<float> distrib(-1.0f, 1.0f);

    for (auto l = 0; l < layers.size(); ++l) {
        auto fan_in = (l == 0) ? static_cast<float>(input_size) : static_cast<float>(layers[l - 1].num_rows);
        auto fan_out = (l == layers.size() - 1) ? 0.0f : static_cast<float>(layers[l + 1].num_rows);
        float value = std::sqrt(6.0f / (fan_in + fan_out));
        for (auto x = 0; x < layers[l].num_rows * layers[l].num_cols; ++x) {
            layers[l].weights[x] = distrib(generator) * value;
        }
    }


}