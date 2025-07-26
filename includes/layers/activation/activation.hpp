#pragma once
#include <cmath>
#include <algorithm>
#include <string>
#include "../core/concepts.hpp"

#ifdef TENSORFLOW_ENABLED
#include <tensorflow/cc/framework/ops.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/cc/ops/standard_ops.h>
#endif

namespace deep_qn {
    namespace layers {

        // ==================== Base Activation Interface ====================
        template<typename T = float>
        class ActivationBase {
        public:
            virtual ~ActivationBase() = default;

#ifdef TENSORFLOW_ENABLED
            virtual tensorflow::Output apply(tensorflow::Scope& scope, tensorflow::Input input) const = 0;
#endif

            virtual T activate(T x) const = 0;
            virtual T derivative(T x) const = 0;
            virtual std::string name() const = 0;
            virtual bool is_differentiable() const { return true; }
            virtual bool has_upper_bound() const { return false; }
            virtual bool has_lower_bound() const { return false; }
            virtual T upper_bound() const { return std::numeric_limits<T>::max(); }
            virtual T lower_bound() const { return std::numeric_limits<T>::lowest(); }
        };

        // ==================== ReLU Activation ====================
        template<typename T = float>
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

            std::string name() const override { return "ReLU"; }
            bool has_lower_bound() const override { return true; }
            T lower_bound() const override { return T(0); }
        };

        // ==================== Leaky ReLU Activation ====================
        template<typename T = float>
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

            T get_alpha() const { return alpha; }
            void set_alpha(T new_alpha) { alpha = new_alpha; }
        };

        // ==================== ELU Activation ====================
        template<typename T = float>
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

            std::string name() const override {
                return "ELU(alpha=" + std::to_string(alpha) + ")";
            }

            bool has_lower_bound() const override { return true; }
            T lower_bound() const override { return -alpha; }

            T get_alpha() const { return alpha; }
            void set_alpha(T new_alpha) { alpha = new_alpha; }
        };

        // ==================== Tanh Activation ====================
        template<typename T = float>
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

            std::string name() const override { return "Tanh"; }
            bool has_upper_bound() const override { return true; }
            bool has_lower_bound() const override { return true; }
            T upper_bound() const override { return T(1); }
            T lower_bound() const override { return T(-1); }
        };

       