#include <iostream>
#include <random>
#include "tensor (8).h"
#include "nn_dense (5).h"
#include "nn_activation (3).h"
#include "nn_loss (5).h"
#include "nn_optimizer (5).h"
#include "neural_network (4).h"
#include "nn_interfaces (4).h"

// Alias local para Tensor
template<typename T, std::size_t Rank>
using Tensor = utec::algebra::Tensor<T, Rank>;

int main() {
    // Generar datos sintéticos para regresión: y = 2x + 3 + ruido
    const size_t n_samples = 100;
    Tensor<float, 2> X(n_samples, 1);
    Tensor<float, 2> Y(n_samples, 1);

    std::default_random_engine eng(42);
    std::uniform_real_distribution<float> dist_x(-10.0f, 10.0f);
    std::normal_distribution<float> noise(0.0f, 1.0f);

    for (size_t i = 0; i < n_samples; ++i) {
        float x = dist_x(eng);
        X(i, 0) = x;
        Y(i, 0) = 2.0f * x + 3.0f + noise(eng);
    }

    // Inicializadores sencillos para pesos y bias (aleatorios pequeños)
    auto init_weights = [](Tensor<float, 2> &w) {
        std::default_random_engine eng(123);
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        for (auto &val: w) val = dist(eng);
    };
    auto init_bias = [](Tensor<float, 2> &b) {
        for (auto &val: b) val = 0.0f;
    };

    // Crear red neuronal con una capa densa y activación ReLU
    utec::neural_network::NeuralNetwork<float> net;
    net.add_layer(std::make_unique<utec::neural_network::Dense<float>>(1, 10, init_weights, init_bias));
    net.add_layer(std::make_unique<utec::neural_network::ReLU<float>>());
    net.add_layer(std::make_unique<utec::neural_network::Dense<float>>(10, 1, init_weights, init_bias));

    // Parámetros de entrenamiento
    const size_t epochs = 100;
    const size_t batch_size = 10;
    const float learning_rate = 0.01f;

    // Mostrar pérdida antes de entrenar
    auto pred_before = net.predict(X);
    utec::neural_network::MSELoss<float> loss_before(pred_before, Y);
    std::cout << "Loss antes de entrenar: " << loss_before.loss() << "\n";

    // Entrenamiento con impresión de pérdida cada 10 épocas
    for (size_t e = 1; e <= epochs; ++e) {
        net.train<utec::neural_network::MSELoss>(X, Y, 1, batch_size, learning_rate);
        if (e % 10 == 0) {
            auto pred = net.predict(X);
            utec::neural_network::MSELoss<float> loss(pred, Y);
            std::cout << "Época " << e << ", Loss: " << loss.loss() << "\n";
        }
    }

    // Mostrar pérdida después de entrenar
    auto pred_after = net.predict(X);
    utec::neural_network::MSELoss<float> loss_after(pred_after, Y);
    std::cout << "Loss después de entrenar: " << loss_after.loss() << "\n";

    // Mostrar algunas predicciones vs valores reales
    std::cout << "\nAlgunas predicciones vs valores reales:\n";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << "X: " << X(i, 0)
                  << ", Pred: " << pred_after(i, 0)
                  << ", Real: " << Y(i, 0) << "\n";
    }

    return 0;

}