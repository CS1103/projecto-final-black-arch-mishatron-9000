//
// Created by Usuario on 22/06/2025.
//

#ifndef EPIC1_OFICIAL_NN_INTERFACES_H
#define EPIC1_OFICIAL_NN_INTERFACES_H
#pragma once

#include <cstddef>
#include <memory>
#include "tensor (8).h"

namespace utec::neural_network {


    template<typename T>
    class IOptimizer;


    template<typename T>
    class ILayer {
    public:
        virtual ~ILayer() = default;
        virtual utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>& input) = 0;
        virtual utec::algebra::Tensor<T,2> backward(const utec::algebra::Tensor<T,2>& grad_output) = 0;
        // Now that IOptimizer is forward‐declared, this compiles
        virtual void update_params(IOptimizer<T>& optimizer) {}
    };


    template<typename T, std::size_t Rank = 2>
    class ILoss {
    public:
        virtual ~ILoss() = default;
        virtual T loss() const = 0;
        virtual utec::algebra::Tensor<T,2> loss_gradient() const = 0;
    };


    template<typename T>
    class IOptimizer {
    public:
        virtual ~IOptimizer() = default;
        virtual void update(utec::algebra::Tensor<T,2>& params,
                            const utec::algebra::Tensor<T,2>& grads) = 0;
    };

}
#endif //EPIC1_OFICIAL_NN_INTERFACES_H
