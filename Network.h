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
    void back_propagate(float *act_output) {
        {
            Layer &current = get_output_layer();
            const size_t out_neurons = current.num_rows;

            std::unique_ptr<float[]> loss_deriv = std::make_unique<float[]>(out_neurons);
            //Computing the derivative of the loss function with respect to the net_output
            Loss::apply_deriv(current.post_activ.get(), current.post_activ.get() + out_neurons,
                              act_output, loss_deriv.get());


            current.derivative(current.pre_activ.get(), current.deltas.get());
            std::transform(current.deltas.get(), current.deltas.get() + out_neurons, loss_deriv.get(),
                           current.deltas.get(), std::multiplies<float>{});

        }

        //stuff for all the other layers
        for (int l = layers.size() - 2; l >= 0; --l) {
            Layer &last = layers[l + 1];
            Layer &current = layers[l];

            std::unique_ptr<float[]> deriv_out = std::make_unique<float[]>(current.num_rows);
            //
            current.derivative(current.pre_activ.get(), deriv_out.get());


            //now we can compute the deltas of all the other layers
            cblas_sgemv(CblasRowMajor, CblasTrans, last.num_rows, last.num_cols, 1.0f, last.weights.get(),
                        last.num_cols, last.deltas.get(),
                        1, 0.0f, current.deltas.get(), 1);


            std::transform(current.deltas.get(), current.deltas.get() + current.num_rows, deriv_out.get(),
                           current.deltas.get(), std::multiplies<float>{});


        }
    }


public:

    Network(size_t input) : input_size(input) {}

    template<typename Activ>
    void add_layer(size_t num_hidden) {
        Layer layer;
        if (layers.empty()) {
            layer.num_cols = input_size;
        } else {
            const Layer &last = layers.back();
            layer.num_cols = last.num_rows;
        }
        layers.emplace_back(std::move(layer));
        Layer &current = layers.back();


        current.num_rows = num_hidden;
        current.weights = std::make_unique<float[]>(current.num_rows * current.num_cols);
        current.biases = std::make_unique<float[]>(current.num_rows);
        current.pre_activ = std::make_unique<float[]>(current.num_rows);
        current.post_activ = std::make_unique<float[]>(current.num_rows);
        current.deltas = std::make_unique<float[]>(current.num_rows);
        //init everything but weights to zero
        std::fill(current.deltas.get(), current.deltas.get() + current.num_rows, 0.0f);
        std::fill(current.biases.get(), current.biases.get() + current.num_rows, 0.0f);


        const int rows = current.num_rows;

        float *in = current.pre_activ.get();
        float *out = current.post_activ.get();
        current.activation = [rows, in, out]() {
            Activ::apply(in, in + rows, out);
        };

        current.derivative = [rows](float *in, float *out) {
            Activ::apply_deriv(in, in + rows, out);
        };
    }

    void init_weights();


    template<typename Loss>
    void train(Train::Data &data, float lr, size_t epoch) {
        std::unique_ptr<float[]> input = std::make_unique<float[]>(input_size);
        std::unique_ptr<float[]> output = std::make_unique<float[]>(get_output_layer().num_rows);
        for (auto i = 0; i < epoch; ++i) {
            std::shuffle(data.mutable_data()->begin(), data.mutable_data()->end(), generator);

            for (auto k = 0; k < data.data_size(); ++k) {
                const Train::Point &point = data.data(k);
                for (auto p = 0; p < input_size; ++p) {
                    input[p] = point.input(p);
                }
                for (auto p = 0; p < get_output_layer().num_rows; ++p) {
                    output[p] = point.output(p);
                }
                forward_pass(input.get());

                back_propagate<Loss>(output.get());

                for (Layer &layer : layers) {
                    for (auto x = 0; x < layer.num_rows; ++x) {
                        layer.biases[x] -= lr * layer.deltas[x];
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
