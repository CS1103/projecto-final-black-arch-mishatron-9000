//
// Created by Usuario on 22/06/2025.
//

#ifndef EPIC1_OFICIAL_NEURAL_NETWORK_H
#define EPIC1_OFICIAL_NEURAL_NETWORK_H

#include "nn_interfaces (4).h"
#include "nn_optimizer (5).h"
#include "nn_loss (5).h"
#include <vector>
#include <memory>

namespace utec::neural_network {

    template<typename T>
    class NeuralNetwork {
        std::vector<std::unique_ptr<ILayer<T>>> layers_;
    public:
        void add_layer(std::unique_ptr<ILayer<T>> layer) {
            layers_.push_back(std::move(layer));
        }

        template <template <typename...> class LossType,
                template <typename...> class OptimizerType = SGD>
        void train(const utec::algebra::Tensor<T,2>& X,
                   const utec::algebra::Tensor<T,2>& Y,
                   size_t epochs, size_t batch_size, T learning_rate) {
            OptimizerType<T> optimizer(learning_rate);
            for (size_t e = 0; e < epochs; ++e) {
                auto a = X;
                for (auto& layer : layers_)
                    a = layer->forward(a);
                LossType<T> loss_obj(a, Y);
                auto grad = loss_obj.loss_gradient();
                for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
                    grad = (*it)->backward(grad);
                for (auto& layer : layers_)
                    layer->update_params(optimizer);
            }
        }

        utec::algebra::Tensor<T,2> predict(const utec::algebra::Tensor<T,2>& X) {
            auto a = X;
            for (auto& layer : layers_)
                a = layer->forward(a);
            return a;
        }
    };

}
#endif //EPIC1_OFICIAL_NEURAL_NETWORK_H
