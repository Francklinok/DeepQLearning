#pragma once
#include "concept.hpp"
#include <vector>
#include <cstdint>
#include <algorithm> // for std::fill

namespace normalization {

    template <FloatingPoint T>
    struct alignas(constants::CACHE_LINE_SIZE) NormalizationData {
        // Main data aligned for cache efficiency
        std::vector<T> input_data;
        std::vector<T> output_data;
        std::vector<T> mean_cache;
        std::vector<T> variance_cache;
        std::vector<T> scale_factors;
        std::vector<T> offset_factors;

        struct Config {
            uint32_t batch_size;
            uint32_t feature_size;
            T epsilon;
            bool use_global_stats;

            Config() noexcept
                : batch_size(0), feature_size(0), epsilon(T{ 1e-5 }), use_global_stats(false) {
            }
        } config;

        NormalizationData() = default;

        // Pre-allocate memory for expected batch and feature sizes
        void reserve_memory(size_t batch_size, size_t feature_size) noexcept {
            const size_t total_elements = batch_size * feature_size;
            input_data.reserve(total_elements);
            output_data.reserve(total_elements);
            mean_cache.reserve(feature_size);
            variance_cache.reserve(feature_size);
            scale_factors.reserve(feature_size);
            offset_factors.reserve(feature_size);
        }

        // Resize all vectors to fit a new batch and feature configuration
        void resize_for_batch(uint32_t new_batch_size, uint32_t new_feature_size) {
            config.batch_size = new_batch_size;
            config.feature_size = new_feature_size;

            const size_t total_elements = new_batch_size * new_feature_size;

            input_data.resize(total_elements);
            output_data.resize(total_elements);
            mean_cache.resize(new_feature_size);
            variance_cache.resize(new_feature_size);

            if (!scale_factors.empty()) {
                scale_factors.resize(new_feature_size, T{ 1 });
            }
            if (!offset_factors.empty()) {
                offset_factors.resize(new_feature_size, T{ 0 });
            }
        }

        // Clear cached mean and variance values
        void clear_caches() noexcept {
            std::fill(mean_cache.begin(), mean_cache.end(), T{ 0 });
            std::fill(variance_cache.begin(), variance_cache.end(), T{ 0 });
        }

        // Check whether all buffers are correctly sized and ready
        bool is_valid() const noexcept {
            const size_t expected_total = static_cast<size_t>(config.batch_size) * config.feature_size;
            return input_data.size() == expected_total &&
                output_data.size() == expected_total &&
                mean_cache.size() == config.feature_size &&
                variance_cache.size() == config.feature_size;
        }

        struct PerformanceStats {
            double total_time_us;
            double avg_time_per_iteration_us;
            double throughput_elements_per_second;
            size_t iterations;
            size_t batch_size;
            size_t feature_size;

            PerformanceStats() noexcept
                : total_time_us(0), avg_time_per_iteration_us(0),
                throughput_elements_per_second(0), iterations(0),
                batch_size(0), feature_size(0) {
            }
        };
    };

}
