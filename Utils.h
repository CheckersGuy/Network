//
// Created by root on 14.03.20.
//

#ifndef NETWORK_UTILS_H
#define NETWORK_UTILS_H

#include <algorithm>
#include <cmath>


namespace Utils {

    struct HyperBolic {
        template<typename Iter, typename Out>
        static void apply(Iter begin, Iter end, Out out) {
            using Type = typename std::iterator_traits<Iter>::value_type;
            static_assert(std::is_same<float, Type>::value);
            std::transform(begin, end, out, [](float value) {
                return std::tanh(value);
            });
        }


        template<typename Iter, typename Out>
        static void apply_deriv(Iter begin, Iter end, Out out) {
            using Type = typename std::iterator_traits<Iter>::value_type;
            static_assert(std::is_same<float, Type>::value);
            std::transform(begin, end, out, [](float value) {
                float sig = std::tanh(value);
                return 1.0f - sig * sig;
            });
        }
    };


    struct Sigmoid {
        template<typename Iter, typename Out>
        static void apply(Iter begin, Iter end, Out out) {
            using Type = typename std::iterator_traits<Iter>::value_type;
            static_assert(std::is_same<float, Type>::value);
            std::transform(begin, end, out, [](float value) {
                float ret = 1.0f / (1.0f + std::exp(-value));
                return ret;
            });
        }


        template<typename Iter, typename Out>
        static void apply_deriv(Iter begin, Iter end, Out out) {
            using Type = typename std::iterator_traits<Iter>::value_type;
            static_assert(std::is_same<float, Type>::value);
            std::transform(begin, end, out, [](float value) {
                float sig = 1.0f / (1.0f + std::exp(-value));
                return sig * (1.0f - sig);
            });
        }

    };


    struct SoftMax {
        template<typename Iter, typename Out>
        static void apply(Iter begin, Iter end, Out out) {
            using Type = typename std::iterator_traits<Iter>::value_type;
            static_assert(std::is_same<float, Type>::value);

            auto distance = end - begin;
            float max_value = *std::max_element(begin, end);
            float summe = 0.0f;
            std::transform(begin, end, out, [&summe, max_value](float value) {
                float ex = std::exp(value - max_value);
                summe += ex;
                return ex;
            });

            std::transform(out, out + distance, out, [summe](float value) {
                return value / summe;
            });

        }


        template<typename Iter, typename Out>
        static void apply_deriv(Iter begin, Iter end, Out out) {

        }
    };

    struct Relu {
        template<typename Iter, typename Out>
        static void apply(Iter begin, Iter end, Out out) {
            using Type = typename std::iterator_traits<Iter>::value_type;
            static_assert(std::is_same<float, Type>::value);
            std::transform(begin, end, out, [](float value) {
                return std::max(0.0f, value);
            });
        }


        template<typename Iter, typename Out>
        static void apply_deriv(Iter begin, Iter end, Out out) {
            using Type = typename std::iterator_traits<Iter>::value_type;
            static_assert(std::is_same<float, Type>::value);
            std::transform(begin, end, out, [](float value) {
                return (value > 0.0f) ? 1.0f : 0.0f;
            });
        }

    };

    struct SquareLoss {
        template<typename Iter, typename Iter2>
        static float apply(Iter net_output, Iter net_end, Iter2 act_output) {
            float summe = 0;
            auto out_iter = act_output;
            for (auto it = net_output; it != net_end;
                 ++it) {
                float net = *it;
                float out = *out_iter;
                summe += (net - out) * (net - out);
                out_iter++;
            }
            return summe / 2.0f;
        }

        template<typename Iter, typename Iter2, typename OutIter>
        static void apply_deriv(Iter net_output, Iter net_end, Iter2 act_output, OutIter out) {
            auto a_out = act_output;
            for (auto it = net_output; it != net_end; ++it) {
                float p = *a_out;
                float y = *it;
                *out = y - p;
                out++;
                a_out++;
            }
        }


    };

    struct CrossLoss {
        template<typename Iter, typename Iter2>
        static float apply(Iter net_output, Iter net_end, Iter2 act_output) {
            float summe = 0;
            auto out_iter = act_output;
            for (auto it = net_output; it != net_end;
                 ++it) {
                float net = *it;
                float out = *out_iter;
                summe += (net - out) * (net - out);
                out_iter++;
            }
            return summe / 2.0f;
        }

        template<typename Iter, typename Iter2, typename OutIter>
        static void apply_deriv(Iter net_output, Iter net_end, Iter2 act_output, OutIter out) {
            auto a_out = act_output;
            for (auto it = net_output; it != net_end; ++it) {
                float p = *a_out;
                float y = *it;
                *out = y - p;
                out++;
                a_out++;
            }
        }

    };

}

#endif //NETWORK_UTILS_H
