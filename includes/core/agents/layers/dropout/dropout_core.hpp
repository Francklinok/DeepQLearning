#pragma once // Prevent multiple inclusion

// Project-specific includes
#include "dropout_concept.hpp"       // For FloatingPoint concept
#include "dropout_data_structur.hpp" // For DropoutData struct
#include "basic_interface.hpp"       // For DropoutBase interface

// Standard library includes
#include <vector>        // For std::vector
#include <string>        // For std::string
#include <span>          // For std::span (C++20)
#include <mutex>         // For std::mutex
#include <random>        // For std::mt19937, distributions
#include <thread>        // For std::thread::hardware_concurrency
#include <atomic>        // For std::atomic
#include <stdexcept>     // For std::invalid_argument
#include <algorithm>     // For std::for_each
#include <execution>     // For std::execution::par_unseq
#include <ranges>        // For std::views::iota
#include <cmath>         // For std::sqrt, std::max

#ifdef TENSORFLOW_ENABLE
#include <tensorflow/cc/framework/ops.h>   // For tensorflow::ops
#include <tensorflow/core/framework/tensor.h>
#endif

namespace dropout {

    // -------------------------
    // Standard Dropout Layer
    // -------------------------
    template <FloatingPoint T>
    class StandardDropout : public DropoutBase<T> {
    private:
        mutable DropoutData<T> data_;   // Stores dropout parameters and RNGs
        mutable std::mutex rng_mutex_;  // Mutex for RNG access in multithreaded contexts

        // Retrieves a thread-specific RNG
        std::mt19937& get_thread_rng() const {
            thread_local static size_t thread_id =
                data_.access_counter.fetch_add(1) % data_.generators.size();
            return data_.generators[thread_id];
        }

    public:
        // Constructor: initializes dropout rate and internal data
        explicit StandardDropout(T dropout_rate = T(0.5))
            : data_(0, dropout_rate) {
            static_assert(T(0) <= dropout_rate && dropout_rate < T(1),
                "Dropout rate must be in [0,1)");
        }

#ifdef TENSORFLOW_ENABLE
        // TensorFlow graph dropout application
        tensorflow::Output apply(tensorflow::Scope& scope,
            tensorflow::Input input,
            bool training = true) const override {
            if (!training) {
                return tensorflow::ops::Identity(scope, input);
            }
            return tensorflow::ops::Dropout(scope, input,
                tensorflow::ops::Dropout::KeepProb(T(1) - data_.dropout_rate));
        }
#endif

        // Applies dropout (single-threaded)
        void apply_dropout(std::span<T> data, bool training = true) override {
            if (!training) return;

            data_.resize(data.size());
            auto& rng = get_thread_rng();
            std::uniform_real_distribution<T> dist(T(0), T(1));

            for (size_t i = 0; i < data.size(); ++i) {
                if (dist(rng) < data_.dropout_rate) {
                    data[i] = T(0);
                    data_.mask[i] = false;
                }
                else {
                    data[i] *= data_.scale_factor;
                    data_.mask[i] = true;
                }
            }
        }

        // Applies dropout in parallel using execution policies
        void apply_dropout_parallel(std::span<T> data, bool training = true) override {
            if (!training) return;
            data_.resize(data.size());

            auto indices = std::views::iota(size_t{ 0 }, data.size());
            std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
                [this, &data](size_t i) {
                    auto& rng = get_thread_rng();
                    std::uniform_real_distribution<T> dist(T(0), T(1));
                    if (dist(rng) < data_.dropout_rate) {
                        data[i] = T(0);
                        data_.mask[i] = false;
                    }
                    else {
                        data[i] *= data_.scale_factor;
                        data_.mask[i] = true;
                    }
                });
        }

        // Forward pass: returns processed data
        std::vector<T> forward(const std::vector<T>& input, bool training = true) override {
            std::vector<T> output = input;
            apply_dropout_parallel(std::span<T>(output), training);
            return output;
        }

        // Name of the dropout instance
        std::string name() const override {
            return "Dropout(rate=" + std::to_string(data_.dropout_rate) + ")";
        }

        // Set dropout rate and update scale factor
        void set_dropout_rate(T rate) override {
            if (rate < T(0) || rate >= T(1)) {
                throw std::invalid_argument("Dropout rate must be in [0,1)");
            }
            data_.dropout_rate = rate;
            data_.update_scale_factor();
        }

        // Set training mode
        void set_training_mode(bool training) override {
            data_.is_training = training;
            data_.update_scale_factor();
        }

        // Check if in training mode
        bool is_training() const override { return data_.is_training; }

        // Retrieve current dropout mask
        const std::vector<bool>& get_mask() const { return data_.mask; }

        // Apply the inverse of dropout to gradients
        void apply_inverse_dropout(std::span<T> gradients) {
            for (size_t i = 0; i < gradients.size() && i < data_.mask.size(); ++i) {
                if (!data_.mask[i]) {
                    gradients[i] = T(0);
                }
                else {
                    gradients[i] *= data_.scale_factor;
                }
            }
        }
    };

    // -------------------------
    // Gaussian Dropout Layer
    // -------------------------
    template <FloatingPoint T = float>
    class GaussianDropout : public DropoutBase<T> {
    private:
        mutable DropoutData<T> data_; // Dropout parameters and RNGs
        T variance_;                  // Variance for Gaussian noise

        // Thread-specific RNG
        std::mt19937& get_thread_rng() const {
            thread_local static size_t thread_id =
                data_.access_counter.fetch_add(1) % data_.generators.size();
            return data_.generators[thread_id];
        }

    public:
        explicit GaussianDropout(T dropout_rate = T(0.5))
            : data_(0, dropout_rate),
            variance_(dropout_rate / (T(1) - dropout_rate)) {
        }

#ifdef TENSORFLOW_ENABLE
        // TensorFlow graph Gaussian dropout (placeholder implementation)
        tensorflow::Output apply(tensorflow::Scope& scope,
            tensorflow::Input input,
            bool training = true) const override {
            if (!training) {
                return tensorflow::ops::Identity(scope, input);
            }
            return tensorflow::ops::Identity(scope, input); // No native Gaussian op
        }
#endif

        // Apply Gaussian noise to data (single-threaded)
        void apply_dropout(std::span<T> data, bool training = true) override {
            if (!training) return;

            auto& rng = get_thread_rng();
            std::normal_distribution<T> dist(T(1), std::sqrt(variance_));

            for (size_t i = 0; i < data.size(); ++i) {
                data[i] *= std::max(T(0), dist(rng));
            }
        }

        // Apply Gaussian noise in parallel
        void apply_dropout_parallel(std::span<T> data, bool training = true) override {
            if (!training) return;

            auto indices = std::views::iota(size_t{ 0 }, data.size());
            std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
                [this, &data](size_t i) {
                    auto& rng = get_thread_rng();
                    std::normal_distribution<T> dist(T(1), std::sqrt(variance_));
                    data[i] *= std::max(T(0), dist(rng));
                });
        }

        // Forward pass
        std::vector<T> forward(const std::vector<T>& input, bool training = true) override {
            std::vector<T> output = input;
            apply_dropout_parallel(std::span<T>(output), training);
            return output;
        }

        // Name of this Gaussian dropout
        std::string name() const override {
            return "GaussianDropout(rate=" + std::to_string(data_.dropout_rate) + ")";
        }

        // Get current dropout rate
        T get_dropout_rate() const override { return data_.dropout_rate; }

        // Set dropout rate and recalculate variance
        void set_dropout_rate(T rate) override {
            if (rate < T(0) || rate >= T(1)) {
                throw std::invalid_argument("Dropout rate must be in [0,1)");
            }
            data_.dropout_rate = rate;
            variance_ = rate / (T(1) - rate);
            data_.update_scale_factor();
        }

        // Set training mode
        void set_training_mode(bool training) override {
            data_.is_training = training;
            data_.update_scale_factor();
        }

        // Check if in training mode
        bool is_training() const override { return data_.is_training; }
    };

} // namespace dropout
