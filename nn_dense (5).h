//
// Created by Usuario on 22/06/2025.
//

#ifndef EPIC1_OFICIAL_NN_DENSE_H
#define EPIC1_OFICIAL_NN_DENSE_H

#include "nn_interfaces.h"
#include "tensor.h"
#include "nn_optimizer.h"
#include <numeric>


template<typename T, std::size_t Rank>
using Tensor = utec::algebra::Tensor<T,Rank>;

namespace utec::neural_network {

    template<typename T>
    class Dense final : public ILayer<T> {
        size_t in_f_, out_f_;
        Tensor<T,2> weights_, bias_;
        Tensor<T,2> last_input_;
        Tensor<T,2> grad_w_, grad_b_;
    public:
        template<typename InitWFun, typename InitBFun>
        Dense(size_t in_f, size_t out_f, InitWFun init_w_fun, InitBFun init_b_fun)
                : in_f_(in_f), out_f_(out_f), weights_(in_f,out_f), bias_(1,out_f) {
            init_w_fun(weights_);
            init_b_fun(bias_);
        }

        Tensor<T,2> forward(const Tensor<T,2>& x) override {
            last_input_ = x;
            auto z = matrix_product(x, weights_);
            for (size_t i = 0; i < z.shape()[0]; ++i)
                for (size_t j = 0; j < z.shape()[1]; ++j)
                    z(i,j) += bias_(0,j);
            return z;
        }

        Tensor<T,2> backward(const Tensor<T,2>& dZ) override {
            auto xT = last_input_.transpose_2d();
            grad_w_ = matrix_product(xT, dZ);
            grad_b_ = Tensor<T,2>(1, out_f_);
            grad_b_.fill(T(0));
            for (size_t i = 0; i < dZ.shape()[0]; ++i)
                for (size_t j = 0; j < dZ.shape()[1]; ++j)
                    grad_b_(0,j) += dZ(i,j);
            return matrix_product(dZ, weights_.transpose_2d());
        }

        void update_params(IOptimizer<T>& optimizer) override {
            optimizer.update(weights_, grad_w_);
            optimizer.update(bias_, grad_b_);
        }
    };

}


#endif //EPIC1_OFICIAL_NN_DENSE_H
