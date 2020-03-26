//
// Created by root on 14.03.20.
//

#ifndef NETWORK_NETWORK_H
#define NETWORK_NETWORK_H

#include <memory>
#include <vector>
#include <functional>
#include <cblas.h>
#include <iostream>
#include <random>
#include "cmake-build-debug/data.pb.h"
#include "Utils.h"

extern std::mt19937 generator;

struct Layer {
    std::unique_ptr<float[]> weights;
    std::unique_ptr<float[]> biases;
    std::unique_ptr<float[]> pre_activ;
    std::unique_ptr<float[]> post_activ;
    std::unique_ptr<float[]> deltas;
    size_t num_rows;
    size_t num_cols;
    std::function<void()> activation;
    std::function<void(float *, float *)> derivative;

    void forward_pass(float *input);

};


class Network {

private:

    std::vector<Layer> layers;
    size_t input_size;

    //looking over backprop implementation
    //Something is not working correctly
    template<typename Loss>
    void back_propagate(float *act_output, float *temp_loss) {
        const size_t out_neurons = get_output_layer().num_rows;


        get_output_layer().derivative(get_output_layer().pre_activ.get(), get_output_layer().deltas.get());
        Loss::apply_deriv(get_output_layer().post_activ.get(), get_output_layer().post_activ.get() + out_neurons,
                          act_output,
                          temp_loss);
        //need to add a special case for softmax+cross_entropy
        for (auto i = 0; i < out_neurons; ++i) {
            get_output_layer().deltas[i] = get_output_layer().deltas[i] * temp_loss[i];
        }
        //updating all the other deltas
        for (int k = layers.size() - 2; k >= 0; k--) {
            Layer &previous = layers[k + 1];
            Layer &current = layers[k];
            const size_t rows = previous.num_rows;
            const size_t cols = previous.num_cols;

            cblas_sgemv(CblasRowMajor, CblasTrans, rows, cols, 1.0f, previous.weights.get(), cols,
                        previous.deltas.get(), 1, 0.0f,
                        current.deltas.get(), 1);
            //saving temporarily
            current.derivative(current.pre_activ.get(), temp_loss);
            //multiply with deltas
            for (auto x = 0; x < current.num_rows; ++x) {
                current.deltas[x] = temp_loss[x] * current.deltas[x];
            }


        }

    }


public:

    Network(size_t input) : input_size(input) {}

    template<typename Activ>
    void add_layer(size_t num_hidden) {
        Layer layer;
        bool is_first_layer = layers.empty();
        if (is_first_layer) {
            layer.num_cols = input_size;
        } else {
            const Layer &last = layers.back();
            layer.num_cols = last.num_rows;
        }
        layer.num_rows = num_hidden;
        layer.weights = std::make_unique<float[]>(layer.num_rows * layer.num_cols);
        layer.biases = std::make_unique<float[]>(layer.num_rows);
        layer.pre_activ = std::make_unique<float[]>(layer.num_rows);
        layer.post_activ = std::make_unique<float[]>(layer.num_rows);
        layer.deltas = std::make_unique<float[]>(layer.num_rows);
        //init everything but weights to zero
        std::fill(layer.deltas.get(), layer.deltas.get() + layer.num_rows, 0.0f);
        std::fill(layer.biases.get(), layer.biases.get() + layer.num_rows, 0.0f);


        const int rows = layer.num_rows;

        float *in = layer.pre_activ.get();
        float *out = layer.post_activ.get();
        layer.activation = [rows, in, out]() {
            Activ::apply(in, in + rows, out);
        };

        layer.derivative = [rows](float *in, float *out) {
            Activ::apply(in, in + rows, out);
        };

        //Initializing the weights of the layer
        float input = (layers.empty()) ? static_cast<float>(input_size) : static_cast<float>(layers.back().num_rows);
        float temp = std::sqrt(2.0f * input);
        std::normal_distribution<float> distrib(-10.0f / temp, 10.0f / temp);
        for (auto i = 0; i < layer.num_rows * layer.num_cols; ++i) {
            layer.weights[i] = distrib(generator);
        }


        layers.emplace_back(std::move(layer));
    }


    template<typename Loss>
    void train(Train::Data &data, float lr, size_t epoch) {
        const Layer &max_layer = *std::max_element(layers.begin(), layers.end(), [](auto &l1, auto &l2) {
            return l1.num_rows < l2.num_rows;
        });
        static std::mt19937 ran_gen(23123u);
        std::unique_ptr<float[]> input = std::make_unique<float[]>(input_size);
        std::unique_ptr<float[]> output = std::make_unique<float[]>(get_output_layer().num_rows);
        std::unique_ptr<float[]> temp_loss = std::make_unique<float[]>(max_layer.num_rows);

        for (auto i = 0; i < epoch; ++i) {
            std::shuffle(data.mutable_data()->begin(), data.mutable_data()->end(), ran_gen);

            for (auto k = 0; k < data.data_size(); ++k) {
                const Train::Point &point = data.data(k);
                for (auto p = 0; p < input_size; ++p) {
                    input[p] = point.input(p);
                }
                for (auto p = 0; p < get_output_layer().num_rows; ++p) {
                    output[p] = point.output(p);
                }
                forward_pass(input.get());

                back_propagate<Loss>(output.get(), temp_loss.get());

                for (Layer &layer : layers) {
                    for (auto x = 0; x < layer.num_rows; ++x) {
                        layer.biases[x] -= lr * layer.deltas[i];
                    }
                }

                for (auto x = 0; x < layers.front().num_rows; ++x) {
                    for (auto y = 0; y < input_size; ++y) {

                        layers.front().weights[x * layers.front().num_cols + y] -=
                                lr * layers.front().deltas[x] * input[y];
                    }
                }

                for (auto l = 1; l < layers.size(); ++l) {
                    Layer &layer = layers[l];
                    Layer &previous = layers[l - 1];

                    for (auto x = 0; x < layer.num_rows; ++x) {
                        for (auto y = 0; y < layer.num_cols; ++y) {
                            layer.weights[x * layer.num_cols + y] -= lr * layer.deltas[x] * previous.post_activ[y];
                        }
                    }
                }


            }
        }
    }


    void forward_pass(float *input);

    Layer &get_output_layer();

    const Layer &get_output_layer() const;

    void train(Train::Data &data, float lr, size_t epoch);
};


#endif //NETWORK_NETWORK_H
