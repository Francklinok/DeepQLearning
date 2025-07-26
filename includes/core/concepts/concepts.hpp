#pragma once
#include <type_traits>
#include <concepts>

#ifdef TENSORFLOW_ENABLED
#include <tensorflow/cc/framework/ops.h>
#include <tensorflow/cc/framework/scope.h>
#endif

namespace deep_qn {
    namespace core {

        // ==================== Basic Type Concepts ====================
        template<typename T>
        concept Numeric = std::is_arithmetic_v<T>;

        template<typename T>
        concept FloatingPoint = std::is_floating_point_v<T>;

        template<typename T>
        concept Integral = std::is_integral_v<T>;

        // ==================== Neural Network Concepts ====================
#ifdef TENSORFLOW_ENABLED
        template<typename T>
        concept ActivationFunction = requires(T t, tensorflow::Scope & scope, tensorflow::Input input) {
            { t.apply(scope, input) } -> std::same_as<tensorflow::Output>;
        };

        template<typename T>
        concept Normalizer = requires(T t, tensorflow::Scope & scope, tensorflow::Input input, const std::string & name) {
            { t.normalize(scope, input, name) } -> std::same_as<tensorflow::Output>;
        };

        template<typename T>
        concept NetworkLayer = requires(T t, tensorflow::Scope & scope, tensorflow::Input input, const std::string & name) {
            { t.forward(scope, input, name) } -> std::same_as<tensorflow::Output>;
        };
#else
// Dummy concepts when TensorFlow is not available
        template<typename T>
        concept ActivationFunction = true;

        template<typename T>
        concept Normalizer = true;

        template<typename T>
        concept NetworkLayer = true;
#endif

        // ==================== Threading Concepts ====================
        template<typename T>
        concept ThreadSafe = requires(T t) {
            { t.is_thread_safe() } -> std::same_as<bool>;
        };

        template<typename T>
        concept Lockable = requires(T t) {
            t.lock();
            t.unlock();
            { t.try_lock() } -> std::same_as<bool>;
        };

        // ==================== Container Concepts ====================
        template<typename T>
        concept Resizable = requires(T t, typename T::size_type size) {
            t.resize(size);
            { t.size() } -> std::same_as<typename T::size_type>;
            { t.capacity() } -> std::same_as<typename T::size_type>;
        };

        template<typename T>
        concept Reservable = requires(T t, typename T::size_type size) {
            t.reserve(size);
            { t.capacity() } -> std::same_as<typename T::size_type>;
        };

        // ==================== Factory Concepts ====================
        template<typename T, typename... Args>
        concept Constructible = requires(Args... args) {
            T(args...);
            std::is_constructible_v<T, Args...>;
        };

        template<typename Factory, typename Product, typename... Args>
        concept FactoryPattern = requires(Factory f, Args... args) {
            { f.create(args...) } -> std::same_as<Product>;
        };

    } // namespace core
} // namespace deep_qn