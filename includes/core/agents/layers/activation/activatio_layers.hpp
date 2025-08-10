#pragma once

#include "activation_concept.hpp"  // Ta définition Numeric
#include "activation_base.hpp"     // ActivationBase<T> et ses dérivés

#include <vector>
#include <string>
#include <algorithm>   // std::transform
#include <execution>   // std::execution::par_unseq
#include <functional>  // std::multiplies

namespace Activation {

    template<Numeric T = float>
    class ActivationLayer {
    private:
        std::unique_ptr<ActivationBase<T>> activation_func;
        mutable std::vector<T> temp_buffer; // For intermediate calculations

    public:
        template<typename ActivationType>
        explicit ActivationLayer(std::unique_ptr<ActivationType> func)
            : activation_func(std::move(func)) {
        }

        // Move constructor and assignment
        ActivationLayer(ActivationLayer&& other) noexcept
            : activation_func(std::move(other.activation_func)),
            temp_buffer(std::move(other.temp_buffer)) {
        }

        ActivationLayer& operator=(ActivationLayer&& other) noexcept {
            if (this != &other) {
                activation_func = std::move(other.activation_func);
                temp_buffer = std::move(other.temp_buffer);
            }
            return *this;
        }

        // Disable copy to prevent expensive operations
        ActivationLayer(const ActivationLayer&) = delete;
        ActivationLayer& operator=(const ActivationLayer&) = delete;

        // Forward pass
        void forward(const std::vector<T>& input, std::vector<T>& output) {
            output.resize(input.size());
            activation_func->activate_batch(input.data(), output.data(), input.size());
        }

        // In-place forward pass
        void forward_inplace(std::vector<T>& data) {
            activation_func->activate_inplace(data.data(), data.size());
        }

        // Backward pass (compute gradients)
        void backward(const std::vector<T>& input, const std::vector<T>& grad_output,
            std::vector<T>& grad_input) {
            grad_input.resize(input.size());

            // Ensure temp buffer is large enough
            if (temp_buffer.size() < input.size()) {
                temp_buffer.resize(input.size());
            }

            // Compute derivatives
            activation_func->derivative_batch(input.data(), temp_buffer.data(), input.size());

            // Element-wise multiplication: grad_input = grad_output * derivative
            std::transform(std::execution::par_unseq,
                grad_output.begin(), grad_output.end(),
                temp_buffer.begin(), grad_input.begin(),
                std::multiplies<T>());
        }

        // Accessors
        const ActivationBase<T>& get_activation() const { return *activation_func; }
        std::string get_name() const { return activation_func->name(); }
    };

} // namespace Activation
