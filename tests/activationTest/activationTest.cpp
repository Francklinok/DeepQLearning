#pragma once

#include <iostream>
#include <vector>
#include <memory>      // std::unique_ptr
#include <string>
#include "activation.hpp"       // Contient toutes les classes d'activation et ActivationFactory
#include "activation_layer.hpp" // Contient ActivationLayer (ton wrapper avec unique_ptr)
#include "activation_concept.hpp"
#include  "includes/core/agent/layers/activation/activation.hpp"

namespace Examples {

    // Exemple simple d'utilisation
    template<typename T = float>
    void basic_usage_example() {
        // Créer les fonctions d'activation
        auto relu = std::make_unique<Activation::ReLUActivation<T>>();
        auto leaky_relu = std::make_unique<Activation::LeakyReLUActivation<T>>(T(0.01));

        T x = T(-0.5);
        std::cout << "ReLU(" << x << ") = " << relu->activate(x) << std::endl;
        std::cout << "LeakyReLU(" << x << ") = " << leaky_relu->activate(x) << std::endl;
    }

    // Exemple de traitement par batch
    template<typename T = float>
    void batch_processing_example() {
        auto gelu = std::make_unique<Activation::GELUActivation<T>>();

        std::vector<T> input = { -2.0, -1.0, 0.0, 1.0, 2.0 };
        std::vector<T> output(input.size());

        gelu->activate_batch(input.data(), output.data(), input.size());

        std::cout << "GELU batch results: ";
        for (T val : output) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    // Exemple d'utilisation avec ActivationLayer (wrapper)
    template<typename T = float>
    void layer_usage_example() {
        auto swish_func = std::make_unique<Activation::SwishActivation<T>>(T(1.0));
        Activation::ActivationLayer<T> layer(std::move(swish_func));

        std::vector<T> input = { -1.0, -0.5, 0.0, 0.5, 1.0 };
        std::vector<T> output;
        layer.forward(input, output);

        std::vector<T> grad_output(output.size(), T(1.0)); // Gradient arbitraire
        std::vector<T> grad_input;
        layer.backward(input, grad_output, grad_input);

        std::cout << "Layer: " << layer.get_name() << std::endl;
        std::cout << "Input gradients: ";
        for (T val : grad_input) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

} // namespace Examples
