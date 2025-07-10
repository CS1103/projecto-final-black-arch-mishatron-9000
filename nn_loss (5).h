//
// Created by Usuario on 22/06/2025.
//

#ifndef EPIC1_OFICIAL_NN_LOSS_H
#define EPIC1_OFICIAL_NN_LOSS_H

#pragma once
#include "nn_interfaces (4).h"
#include <cmath>

namespace utec::neural_network {


    template<typename T>
    class MSELoss final : public ILoss<T,2> {
        utec::algebra::Tensor<T,2> y_pred_, y_true_;
    public:
        MSELoss(const utec::algebra::Tensor<T,2>& y_pred,
                const utec::algebra::Tensor<T,2>& y_true)
                : y_pred_(y_pred), y_true_(y_true) {}

        T loss() const override {
            T sum = T(0);
            auto it_p = y_pred_.cbegin(), it_t = y_true_.cbegin();
            while (it_p != y_pred_.cend()) {
                T diff = *it_p++ - *it_t++;
                sum += diff * diff;
            }
            return sum / static_cast<T>(y_pred_.shape()[0] * y_pred_.shape()[1]);
        }

        utec::algebra::Tensor<T,2> loss_gradient() const override {
            auto grad = y_pred_;
            auto it_p = y_pred_.cbegin(), it_t = y_true_.cbegin(), it_g = grad.begin();
            T scale = T(2) / static_cast<T>(y_pred_.shape()[0] * y_pred_.shape()[1]);
            while (it_p != y_pred_.cend()) {
                *it_g++ = (*it_p++ - *it_t++) * scale;
            }
            return grad;
        }
    };


    template<typename T>
    class BCELoss final : public ILoss<T,2> {
        utec::algebra::Tensor<T,2> y_pred_, y_true_;
    public:

        BCELoss(const utec::algebra::Tensor<T,2>& y_pred,
                const utec::algebra::Tensor<T,2>& y_true)
                : y_pred_(y_pred), y_true_(y_true) {}

        T loss() const override {
            T sum = T(0);
            auto it_p = y_pred_.cbegin(), it_t = y_true_.cbegin();
            while (it_p != y_pred_.cend()) {
                T p = *it_p++;
                T y = *it_t++;
                sum += - (y * std::log(p + T(1e-12)) + (T(1)-y) * std::log(T(1)-p + T(1e-12)));
            }
            return sum / static_cast<T>(y_pred_.shape()[0] * y_pred_.shape()[1]);
        }

        utec::algebra::Tensor<T,2> loss_gradient() const override {
            auto grad = y_pred_;
            auto it_p = y_pred_.cbegin(), it_t = y_true_.cbegin(), it_g = grad.begin();
            T inv_n = T(1) / static_cast<T>(y_pred_.shape()[0] * y_pred_.shape()[1]);
            while (it_p != y_pred_.cend()) {
                T p = *it_p++;
                T y = *it_t++;
                *it_g++ = inv_n * (-(y / (p + T(1e-12))) + ((T(1)-y) / (T(1)-p + T(1e-12))));
            }
            return grad;
        }
    };

} // namespace utec::neural_network

#endif //EPIC1_OFICIAL_NN_LOSS_H
