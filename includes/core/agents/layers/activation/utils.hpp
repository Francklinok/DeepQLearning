#pragma once

#include <algorithm>
#include <execution>
#include <cmath>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <memory>
#include <chrono>
#include <stdexcept>
#include <random>
#include <cstddef>

#ifdef __AVX__
#include <immintrin.h>
#endif

namespace details {

    template<typename T>
    inline void vectorized_max(const T* a, const T* b, T* result, size_t size);

#ifdef __AVX__
    template<>
    inline void vectorized_max<float>(const float* a, const float* b, float* result, size_t size) {
        constexpr size_t simd_width = 8;
        if (size == 0) return;

        size_t simd_end = size - (size % simd_width);

        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            __m256 vmax = _mm256_max_ps(va, vb);
            _mm256_storeu_ps(&result[i], vmax);
        }
        for (size_t i = simd_end; i < size; ++i) {
            result[i] = std::max(a[i], b[i]);
        }
    }
#else
    template<>
    inline void vectorized_max<float>(const float* a, const float* b, float* result, size_t size) {
        // fallback scalar implementation
        for (size_t i = 0; i < size; ++i) {
            result[i] = std::max(a[i], b[i]);
        }
    }
#endif

    // For double precision, fallback scalar only
    template<>
    inline void vectorized_max<double>(const double* a, const double* b, double* result, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = std::max(a[i], b[i]);
        }
    }

    template<typename T>
    inline void vectorized_exp(const T* input, T* output, size_t size) {
        std::transform(std::execution::par_unseq, input, input + size, output,
            [](T x) { return std::exp(x); });
    }

    template<typename T>
    inline void vectorized_tanh(const T* input, T* output, size_t size) {
        std::transform(std::execution::par_unseq, input, input + size, output,
            [](T x) { return std::tanh(x); });
    }

    template<Numeric T = float>
    class ActivationBenchmark {
    public:
        struct BenchmarkResult {
            std::string name;
            double forward_time_ms;
            double backward_time_ms;
            double total_time_ms;
        };

        static BenchmarkResult benchmark_activation(
            const ActivationBase<T>& activation,
            size_t data_size = 1000000,
            size_t iterations = 100) {

            std::vector<T> input(data_size);
            std::vector<T> output(data_size);
            std::vector<T> grad_output(data_size);
            std::vector<T> grad_input(data_size);

            // Modern RNG
            std::mt19937 rng(42); // fixed seed for reproducibility
            std::uniform_real_distribution<T> dist(-0.5, 0.5);

            std::generate(input.begin(), input.end(), [&]() { return dist(rng); });
            std::fill(grad_output.begin(), grad_output.end(), T(1));

            auto start = std::chrono::high_resolution_clock::now();

            for (size_t i = 0; i < iterations; ++i) {
                activation.activate_batch(input.data(), output.data(), data_size);
            }

            auto mid = std::chrono::high_resolution_clock::now();

            for (size_t i = 0; i < iterations; ++i) {
                activation.derivative_batch(input.data(), grad_input.data(), data_size);
            }

            auto end = std::chrono::high_resolution_clock::now();

            double forward_time = std::chrono::duration<double, std::milli>(mid - start).count();
            double backward_time = std::chrono::duration<double, std::milli>(end - mid).count();

            return {
                activation.name(),
                forward_time / iterations,
                backward_time / iterations,
                (forward_time + backward_time) / iterations
            };
        }
    };

    template<Numeric T = float>
    class ActivationRegistry {
    private:
        using CreateFunc = std::function<std::unique_ptr<ActivationBase<T>>(T)>;
        std::unordered_map<std::string, CreateFunc> registry;

    public:
        ActivationRegistry() {
            register_activation("relu", [](T) { return std::make_unique<ReLUActivation<T>>(); });
            register_activation("leaky_relu", [](T alpha) { return std::make_unique<LeakyReLUActivation<T>>(alpha); });
            register_activation("elu", [](T alpha) { return std::make_unique<ELUActivation<T>>(alpha); });
            register_activation("tanh", [](T) { return std::make_unique<TanhActivation<T>>(); });
            register_activation("sigmoid", [](T) { return std::make_unique<SigmoidActivation<T>>(); });
            register_activation("swish", [](T beta) { return std::make_unique<SwishActivation<T>>(beta); });
            register_activation("gelu", [](T) { return std::make_unique<GELUActivation<T>>(); });
            register_activation("mish", [](T) { return std::make_unique<MishActivation<T>>(); });
        }

        void register_activation(const std::string& name, CreateFunc creator) {
            registry[name] = std::move(creator);
        }

        std::unique_ptr<ActivationBase<T>> create(const std::string& name, T param = T(1.0)) const {
            auto it = registry.find(name);
            if (it != registry.end()) {
                return it->second(param);
            }
            throw std::invalid_argument("Unknown activation: " + name);
        }

        std::vector<std::string> list_activations() const {
            std::vector<std::string> names;
            for (const auto& p : registry) {
                names.push_back(p.first);
            }
            return names;
        }
    };

} // namespace details
