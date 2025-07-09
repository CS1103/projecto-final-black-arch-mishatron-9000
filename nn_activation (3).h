//
// Created by Usuario on 22/06/2025.
//

#ifndef EPIC1_OFICIAL_NN_ACTIVATION_H
#define EPIC1_OFICIAL_NN_ACTIVATION_H
#pragma once
#include "nn_interfaces.h"
#include <cmath>

namespace utec::neural_network {

    template<typename T>
    class ReLU final : public ILayer<T> {
        utec::algebra::Tensor<T,2> last_z_;
    public:
        utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>& z) override {
            last_z_ = z;
            auto out = z;
            for (auto& v : out) v = (v > T(0) ? v : T(0));
            return out;
        }
        utec::algebra::Tensor<T,2> backward(const utec::algebra::Tensor<T,2>& g) override {
            auto grad = g;
            auto it_z = last_z_.cbegin();
            for (auto& v : grad) {
                v = ((*it_z++) > T(0)) ? v : T(0);
            }
            return grad;
        }
    };

    template<typename T>
    class Sigmoid final : public ILayer<T> {
        utec::algebra::Tensor<T,2> last_out_;
    public:
        utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>& z) override {
            auto out = z;
            for (auto& v : out) v = T(1) / (T(1) + std::exp(-v));
            last_out_ = out;
            return out;
        }
        utec::algebra::Tensor<T,2> backward(const utec::algebra::Tensor<T,2>& g) override {
            auto grad = g;
            auto it_o = last_out_.cbegin();
            for (auto& v : grad) {
                T o = *it_o++;
                v = v * o * (T(1) - o);
            }
            return grad;
        }
    };

}

#endif //EPIC1_OFICIAL_NN_ACTIVATION_H
