#pragma once

#include <algorithm>    // std::max, std::transform
#include <execution>    // std::execution::par_unseq
#include <cmath>        // std::exp, std::tanh, std::log
#include <string>       // std::string, std::to_string
#include <type_traits>  // std::is_same_v
#include <cstddef>      // size_t

#include "activation_concept.hpp"  // concept Numeric
#include "activation_base.hpp"     // ActivationBase<T>
#include "vectorized_functions.hpp" // supposé contenir detail::vectorized_max, vectorized_tanh

namespace Activation {

    template <Numeric T = float>
    class ReLUActivation : public ActivationBase<T> {
    public:
        ReLUActivation() = default;

#ifdef TENSORFLOW_ENABLED
        tensorflow::Output apply(tensorflow::Scope& scope, tensorflow::Input input) const override {
            return tensorflow::ops::Relu(scope, input);
        }
#endif

        T activate(T x) const override {
            return std::max(T(0), x);
        }

        T derivative(T x) const override {
            return x > T(0) ? T(1) : T(0);
        }

        void activate_batch(const T* input, T* output, size_t size) const override {
            if constexpr (std::is_same_v<T, float>) {
                const float zero = 0.0f;
                detail::vectorized_max(input, &zero, output, size);
            }
            else {
                ActivationBase<T>::activate_batch(input, output, size);
            }
        }

        std::string name() const override { return "ReLU"; }
        bool has_lower_bound() const override { return true; }
        T lower_bound() const override { return T(0); }
        bool is_monotonic() const override { return true; }
        bool supports_simd() const override { return std::is_same_v<T, float>; }
        bool is_cheap_to_compute() const override { return true; }
    };

    template<Numeric T = float>
    class LeakyReLUActivation : public ActivationBase<T> {
    private:
        T alpha;

    public:
        explicit LeakyReLUActivation(T a = T(0.01)) : alpha(a) {}

#ifdef TENSORFLOW_ENABLED
        tensorflow::Output apply(tensorflow::Scope& scope, tensorflow::Input input) const override {
            return tensorflow::ops::LeakyRelu(scope, input,
                tensorflow::ops::LeakyRelu::Alpha(alpha));
        }
#endif

        T activate(T x) const override {
            return x > T(0) ? x : alpha * x;
        }

        T derivative(T x) const override {
            return x > T(0) ? T(1) : alpha;
        }

        std::string name() const override {
            return "LeakyReLU(alpha=" + std::to_string(alpha) + ")";
        }

        bool is_monotonic() const override { return true; }
        bool supports_simd() const override { return std::is_same_v<T, float>; }
        bool is_cheap_to_compute() const override { return true; }

        T get_alpha() const { return alpha; }
        void set_alpha(T new_alpha) { alpha = new_alpha; }
    };

    template<Numeric T = float>
    class ELUActivation : public ActivationBase<T> {
    private:
        T alpha;

    public:
        explicit ELUActivation(T a = T(1.0)) : alpha(a) {}

#ifdef TENSORFLOW_ENABLED
        tensorflow::Output apply(tensorflow::Scope& scope, tensorflow::Input input) const override {
            return tensorflow::ops::Elu(scope, input);
        }
#endif

        T activate(T x) const override {
            return x > T(0) ? x : alpha * (std::exp(x) - T(1));
        }

        T derivative(T x) const override {
            return x > T(0) ? T(1) : alpha * std::exp(x);
        }

        void activate_batch(const T* input, T* output, size_t size) const override {
            if constexpr (std::is_same_v<T, float>) {
                std::transform(std::execution::par_unseq, input, input + size, output,
                    [this](T x) {
                        return x > T(0) ? x : alpha * (std::exp(x) - T(1));
                    });
            }
            else {
                ActivationBase<T>::activate_batch(input, output, size);
            }
        }

        std::string name() const override {
            return "ELU(alpha=" + std::to_string(alpha) + ")";
        }

        bool has_lower_bound() const override { return true; }
        T lower_bound() const override { return -alpha; }
        bool is_cheap_to_compute() const override { return false; }
        T get_alpha() const { return alpha; }
        void set_alpha(T new_alpha) { alpha = new_alpha; }
    };

    template<Numeric T = float>
    class TanhActivation : public ActivationBase<T> {
    public:
        TanhActivation() = default;

#ifdef TENSORFLOW_ENABLED
        tensorflow::Output apply(tensorflow::Scope& scope, tensorflow::Input input) const override {
            return tensorflow::ops::Tanh(scope, input);
        }
#endif

        T activate(T x) const override {
            return std::tanh(x);
        }

        T derivative(T x) const override {
            T tanh_x = std::tanh(x);
            return T(1) - tanh_x * tanh_x;
        }

        void activate_batch(const T* input, T* output, size_t size) const override {
            if constexpr (std::is_same_v<T, float>) {
                detail::vectorized_tanh(input, output, size);
            }
            else {
                ActivationBase<T>::activate_batch(input, output, size);
            }
        }

        std::string name() const override { return "Tanh"; }
        bool has_upper_bound() const override { return true; }
        bool has_lower_bound() const override { return true; }
        T upper_bound() const override { return T(1); }
        T lower_bound() const override { return T(-1); }
        bool is_monotonic() const override { return true; }
        bool supports_simd() const override { return std::is_same_v<T, float>; }
        bool is_cheap_to_compute() const override { return false; }
    };

    template<Numeric T = float>
    class SigmoidActivation : public ActivationBase<T> {
    public:
        SigmoidActivation() = default;

#ifdef TENSORFLOW_ENABLED
        tensorflow::Output apply(tensorflow::Scope& scope, tensorflow::Input input) const override {
            return tensorflow::ops::Sigmoid(scope, input);
        }
#endif

        T activate(T x) const override {
            return T(1) / (T(1) + std::exp(-x));
        }

        T derivative(T x) const override {
            T sig_x = activate(x);
            return sig_x * (T(1) - sig_x);
        }

        std::string name() const override { return "Sigmoid"; }
        bool has_upper_bound() const override { return true; }
        bool has_lower_bound() const override { return true; }
        T upper_bound() const override { return T(1); }
        T lower_bound() const override { return T(0); }
        bool is_monotonic() const override { return true; }
        bool is_cheap_to_compute() const override { return false; }
    };

    template<Numeric T = float>
    class SwishActivation : public ActivationBase<T> {
    private:
        T beta;

    public:
        explicit SwishActivation(T b = T(1.0)) : beta(b) {}

#ifdef TENSORFLOW_ENABLED
        tensorflow::Output apply(tensorflow::Scope& scope, tensorflow::Input input) const override {
            auto sigmoid = tensorflow::ops::Sigmoid(scope, tensorflow::ops::Mul(scope, input, beta));
            return tensorflow::ops::Mul(scope, input, sigmoid);
        }
#endif

        T activate(T x) const override {
            return x / (T(1) + std::exp(-beta * x));
        }

        T derivative(T x) const override {
            T sigmoid_beta_x = T(1) / (T(1) + std::exp(-beta * x));
            return sigmoid_beta_x + x * sigmoid_beta_x * (T(1) - sigmoid_beta_x) * beta;
        }

        std::string name() const override {
            return "Swish(beta=" + std::to_string(beta) + ")";
        }

        bool has_lower_bound() const override { return true; }
        T lower_bound() const override { return T(0); }
        bool is_cheap_to_compute() const override { return false; }
        T get_beta() const { return beta; }
        void set_beta(T new_beta) { beta = new_beta; }
    };

    template<Numeric T = float>
    class GELUActivation : public ActivationBase<T> {
    private:
        static constexpr T SQRT_2_PI = T(0.7978845608028654);
        static constexpr T COEFF = T(0.044715);

    public:
        GELUActivation() = default;

#ifdef TENSORFLOW_ENABLED
        tensorflow::Output apply(tensorflow::Scope& scope, tensorflow::Input input) const override {
            return tensorflow::ops::Gelu(scope, input);
        }
#endif

        T activate(T x) const override {
            T x_cubed = x * x * x;
            T inner = SQRT_2_PI * (x + COEFF * x_cubed);
            return T(0.5) * x * (T(1) + std::tanh(inner));
        }

        T derivative(T x) const override {
            T x_cubed = x * x * x;
            T inner = SQRT_2_PI * (x + COEFF * x_cubed);
            T tanh_inner = std::tanh(inner);
            T sech2_inner = T(1) - tanh_inner * tanh_inner;
            T inner_derivative = SQRT_2_PI * (T(1) + T(3) * COEFF * x * x);

            return T(0.5) * (T(1) + tanh_inner) + T(0.5) * x * sech2_inner * inner_derivative;
        }

        std::string name() const override { return "GELU"; }
        bool is_cheap_to_compute() const override { return false; }
    };

    template<Numeric T = float>
    class MishActivation : public ActivationBase<T> {
    public:
        MishActivation() = default;

        T activate(T x) const override {
            return x * std::tanh(std::log(T(1) + std::exp(x)));
        }

        T derivative(T x) const override {
            T exp_x = std::exp(x);
            T softplus = std::log(T(1) + exp_x);
            T tanh_softplus = std::tanh(softplus);
            T sech2_softplus = T(1) - tanh_softplus * tanh_softplus;
            T softplus_derivative = exp_x / (T(1) + exp_x);

            return tanh_softplus + x * sech2_softplus * softplus_derivative;
        }

        std::string name() const override { return "Mish"; }
        bool is_cheap_to_compute() const override { return false; }
    };

} // namespace Activation
