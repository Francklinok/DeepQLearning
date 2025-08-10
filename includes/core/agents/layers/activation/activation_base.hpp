#pragma once

#include <iostream>
#include <cmath>          // std::exp, std::tanh, etc.
#include <algorithm>      // std::transform
#include <string>         // std::string, std::numeric_limits
#include <execution>      // std::execution::par_unseq
#include <limits>         // std::numeric_limits
#include <cstddef>        // size_t

#include "activation_concept.hpp" // Pour le concept Numeric

#ifdef TENSORFLOW_ENABLED
#include <tensorflow/cc/framework/ops.h>    // tensorflow::Output, tensorflow::Scope
#include <tensorflow/cc/ops/standard_ops.h> // tensorflow::ops::*
#endif

namespace Activation {

    template<Numeric T = float>
    class ActivationBase {
    public:
        virtual ~ActivationBase() = default;

        // Core interface : fonction d'activation et dérivée
        virtual T activate(T x) const = 0;
        virtual T derivative(T x) const = 0;
        virtual std::string name() const = 0;

        // Batch processing with SIMD optimization (parallélisation)
        virtual void activate_batch(const T* input, T* output, size_t size) const {
            std::transform(std::execution::par_unseq, input, input + size, output,
                [this](T x) { return this->activate(x); });
        }

        virtual void derivative_batch(const T* input, T* output, size_t size) const {
            std::transform(std::execution::par_unseq, input, input + size, output,
                [this](T x) { return this->derivative(x); });
        }

        // Opérations in-place pour optimiser la mémoire
        virtual void activate_inplace(T* data, size_t size) const {
            std::transform(std::execution::par_unseq, data, data + size, data,
                [this](T x) { return this->activate(x); });
        }

        virtual void derivative_inplace(T* data, size_t size) const {
            std::transform(std::execution::par_unseq, data, data + size, data,
                [this](T x) { return this->derivative(x); });
        }

#ifdef TENSORFLOW_ENABLED
        virtual tensorflow::Output apply(tensorflow::Scope& scope, tensorflow::Input input) const = 0;
#endif

        // Propriétés (facultatives) pour indiquer certaines caractéristiques
        virtual bool is_differentiable() const { return true; }
        virtual bool has_upper_bound() const { return false; }
        virtual bool has_lower_bound() const { return false; }
        virtual T upper_bound() const { return std::numeric_limits<T>::max(); }
        virtual T lower_bound() const { return std::numeric_limits<T>::lowest(); }
        virtual bool is_monotonic() const { return false; }
        virtual bool is_continuous() const { return true; }

        // Indications de performance (SIMD, coût calcul)
        virtual bool supports_simd() const { return false; }
        virtual bool is_cheap_to_compute() const { return true; }
    };

}
