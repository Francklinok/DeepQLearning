#pragma once

// Standard library headers
#include <vector>        // std::vector
#include <random>        // std::mt19937, std::random_device, std::uniform_real_distribution
#include <thread>        // std::thread::hardware_concurrency
#include <algorithm>     // std::min, std::for_each
#include <execution>     // std::execution::par_unseq
#include <span>          // std::span
#include <ranges>        // std::views::iota
#include <cstddef>       // size_t




namespace dropout {
    // ---------------------------------------------------------
        // BatchDropoutManager:
        // Handles dropout operations for entire batches in parallel
        // ---------------------------------------------------------
    template <FloatingPoint T>
    class BatchDropoutManager {
    private:
        // Structure to store all batch-related data, aligned for performance
        struct alignas(64) BatchData {
            std::vector<std::vector<T>> batch_data;   // Input data for each batch item
            std::vector<std::vector<bool>> batch_masks; // Mask indicating which elements were kept
            std::vector<T> dropout_rates;             // Per-batch dropout rates
            std::vector<bool> training_flags;         // Per-batch training mode flags
            size_t batch_size{ 0 };                   // Max number of batch items
            size_t feature_size{ 0 };                 // Number of features per batch item
        };

        BatchData batch_data_;               // Storage for all batches
        std::vector<std::mt19937> thread_rngs_; // RNGs for each thread

    public:
        // Constructor initializes batch storage and RNGs
        explicit BatchDropoutManager(size_t max_batch_size, size_t feature_size) {
            batch_data_.batch_size = max_batch_size;
            batch_data_.feature_size = feature_size;

            // Allocate batch storage
            batch_data_.batch_data.resize(max_batch_size, std::vector<T>(feature_size));
            batch_data_.batch_masks.resize(max_batch_size, std::vector<bool>(feature_size));
            batch_data_.dropout_rates.resize(max_batch_size, T(0.5));
            batch_data_.training_flags.resize(max_batch_size, true);

            // Initialize thread-local RNGs
            const auto num_threads = std::thread::hardware_concurrency();
            thread_rngs_.reserve(num_threads);
            std::random_device rd;
            for (size_t i = 0; i < num_threads; ++i) {
                thread_rngs_.emplace_back(rd() + i);
            }
        }

        // Apply dropout to a batch in parallel
        void process_batch_parallel(std::span<std::vector<T>> batch_inputs) {
            const auto batch_size = std::min(batch_inputs.size(), batch_data_.batch_size);
            auto batch_indices = std::views::iota(size_t{ 0 }, batch_size);

            std::for_each(std::execution::par_unseq,
                batch_indices.begin(), batch_indices.end(),
                [this, &batch_inputs](size_t batch_idx) {
                    if (!batch_data_.training_flags[batch_idx]) return; // Skip if not training

                    const size_t thread_id = batch_idx % thread_rngs_.size();
                    auto& rng = thread_rngs_[thread_id];
                    std::uniform_real_distribution<T> dist(T(0), T(1));

                    const T dropout_rate = batch_data_.dropout_rates[batch_idx];
                    const T scale_factor = T(1) / (T(1) - dropout_rate);

                    auto& input = batch_inputs[batch_idx];
                    auto& mask = batch_data_.batch_masks[batch_idx];

                    // Apply dropout element-wise
                    for (size_t i = 0; i < input.size(); ++i) {
                        if (dist(rng) < dropout_rate) {
                            input[i] = T(0);
                            mask[i] = false;
                        }
                        else {
                            input[i] *= scale_factor;
                            mask[i] = true;
                        }
                    }
                });
        }

        // Set dropout rate for a specific batch
        void set_batch_dropout_rate(size_t batch_idx, T rate) {
            if (batch_idx < batch_data_.dropout_rates.size()) {
                batch_data_.dropout_rates[batch_idx] = rate;
            }
        }

        // Set training mode for a specific batch
        void set_batch_training_mode(size_t batch_idx, bool training) {
            if (batch_idx < batch_data_.training_flags.size()) {
                batch_data_.training_flags[batch_idx] = training;
            }
        }
    };
}