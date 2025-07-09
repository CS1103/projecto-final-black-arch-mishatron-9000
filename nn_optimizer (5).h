//
// Created by Usuario on 22/06/2025.
//

#ifndef EPIC1_OFICIAL_NN_OPTIMIZER_H
#define EPIC1_OFICIAL_NN_OPTIMIZER_H
#pragma once
#include "nn_interfaces.h"
#include <cmath>

namespace utec::neural_network {

    template<typename T>
    class SGD final : public IOptimizer<T> {
        T lr_;
    public:
        explicit SGD(T learning_rate = T(0.01)) : lr_(learning_rate) {}
        void update(utec::algebra::Tensor<T,2>& params,
                    const utec::algebra::Tensor<T,2>& grads) override {
            auto it_p = params.begin();
            auto it_g = grads.cbegin();
            while (it_p != params.end() && it_g != grads.cend()) {
                *it_p = *it_p - lr_ * (*it_g);
                ++it_p;
                ++it_g;
            }
        }
    };

    template<typename T>
    class Adam final : public IOptimizer<T> {
        T lr_, beta1_, beta2_, eps_;
        std::size_t t_;
        utec::algebra::Tensor<T,2> m_, v_;
        bool initialized_;
    public:
        explicit Adam(T learning_rate = T(0.001), T beta1 = T(0.9), T beta2 = T(0.999), T epsilon = T(1e-8))
                : lr_(learning_rate), beta1_(beta1), beta2_(beta2), eps_(epsilon),
                  t_(0), initialized_(false) {}

        void update(utec::algebra::Tensor<T,2>& params,
                    const utec::algebra::Tensor<T,2>& grads) override {
            if (!initialized_) {
                m_ = utec::algebra::Tensor<T,2>(params.shape());
                v_ = utec::algebra::Tensor<T,2>(params.shape());
                m_.fill(T(0));
                v_.fill(T(0));
                initialized_ = true;
            }
            ++t_;
            // biased moment estimates
            auto it_m = m_.begin();
            auto it_v = v_.begin();
            auto it_g = grads.cbegin();
            while (it_g != grads.cend()) {
                T g = *it_g;
                *it_m = beta1_ * (*it_m) + (T(1) - beta1_) * g;
                *it_v = beta2_ * (*it_v) + (T(1) - beta2_) * g * g;
                ++it_m;
                ++it_v;
                ++it_g;
            }

            T bc1 = T(1) - std::pow(beta1_, t_);
            T bc2 = T(1) - std::pow(beta2_, t_);
            auto it_p = params.begin();
            it_m = m_.begin();
            it_v = v_.begin();
            while (it_p != params.end()) {
                T m_hat = *it_m / bc1;
                T v_hat = *it_v / bc2;
                *it_p = *it_p - lr_ * m_hat / (std::sqrt(v_hat) + eps_);
                ++it_p;
                ++it_m;
                ++it_v;
            }
        }
    };

}

#endif //EPIC1_OFICIAL_NN_OPTIMIZER_H
