#pragma once

#include "strategy_interface.hpp"
#include "simd_utils.hpp"
#include <execution>
#include <algorithm>
#include <numeric>

namespace normalization {

    // Optimized Batch Normalization implementation using scalar math
    template<FloatingPoint T>
    class BatchNormalizationOptimized final : public NormalizationStrategy<T> {
    private:
        // Compute per-feature mean and variance
        void compute_batch_statistics(const NormalizationData<T>& data,
            std::span<T> means,
            std::span<T> variances) const noexcept {
            const auto& config = data.config;
            const std::span input_span{ data.input_data };

            // Initialize means and variances
            std::fill(std::execution::par_unseq, means.begin(), means.end(), T{ 0 });
            std::fill(std::execution::par_unseq, variances.begin(), variances.end(), T{ 0 });

            // Compute mean per feature
            for (uint32_t batch = 0; batch < config.batch_size; ++batch) {
                for (uint32_t feature = 0; feature < config.feature_size; ++feature) {
                    const size_t idx = batch * config.feature_size + feature;
                    means[feature] += input_span[idx];
                }
            }

            const T inv_batch = T{ 1 } / static_cast<T>(config.batch_size);
            std::transform(std::execution::par_unseq,
                means.begin(), means.end(), means.begin(),
                [inv_batch](T m) { return m * inv_batch; });

            // Compute variance per feature
            for (uint32_t batch = 0; batch < config.batch_size; ++batch) {
                for (uint32_t feature = 0; feature < config.feature_size; ++feature) {
                    const size_t idx = batch * config.feature_size + feature;
                    const T diff = input_span[idx] - means[feature];
                    variances[feature] += diff * diff;
                }
            }

            std::transform(std::execution::par_unseq,
                variances.begin(), variances.end(), variances.begin(),
                [inv_batch](T v) { return v * inv_batch; });
        }

    public:
        void normalize(NormalizationData<T>& data) const noexcept override {
            const auto& config = data.config;

            if (!validate_input(data)) return;

            // Resize mean and variance cache if needed
            if (data.mean_cache.size() != config.feature_size) {
                data.mean_cache.resize(config.feature_size);
                data.variance_cache.resize(config.feature_size);
            }

            // Compute batch-wise statistics
            compute_batch_statistics(data,
                std::span{ data.mean_cache },
                std::span{ data.variance_cache });

            const std::span input_span{ data.input_data };
            const std::span output_span{ data.output_data };

            // Apply normalization per element
            std::transform(std::execution::par_unseq,
                std::views::iota(0u, config.batch_size * config.feature_size).begin(),
                std::views::iota(0u, config.batch_size * config.feature_size).end(),
                output_span.begin(),
                [&](uint32_t idx) -> T {
                    const uint32_t feature_idx = idx % config.feature_size;
                    const T mean = data.mean_cache[feature_idx];
                    const T variance = data.variance_cache[feature_idx];
                    const T inv_std = T{ 1 } / std::sqrt(variance + config.epsilon);

                    const T scale = data.scale_factors.empty() ? T{ 1 } : data.scale_factors[feature_idx];
                    const T offset = data.offset_factors.empty() ? T{ 0 } : data.offset_factors[feature_idx];

                    return scale * (input_span[idx] - mean) * inv_std + offset;
                });
        }

        std::string_view get_name() const noexcept override {
            return "BatchNormalizationOptimized";
        }

        bool requires_global_stats() const noexcept override {
            return true;
        }
    };

    // SIMD-optimized version of Batch Normalization for float
    template<>
    class BatchNormalizationSIMD final : public SimdNormalizationStrategy<float> {
    private:
        // Compute statistics using SIMD
        void compute_statistics_simd(const std::span<const float> input,
            std::span<float> means,
            std::span<float> variances,
            uint32_t batch_size,
            uint32_t feature_size) const noexcept {

            using namespace simd::f32;

            std::fill(std::execution::par_unseq, means.begin(), means.end(), 0.0f);
            std::fill(std::execution::par_unseq, variances.begin(), variances.end(), 0.0f);

            // Compute SIMD mean/variance for each feature
            std::for_each(std::execution::par_unseq,
                std::views::iota(0u, feature_size).begin(),
                std::views::iota(0u, feature_size).end(),
                [&](uint32_t feature_idx) {
                    std::vector<float> feature_values;
                    feature_values.reserve(batch_size);

                    for (uint32_t batch = 0; batch < batch_size; ++batch) {
                        const size_t idx = batch * feature_size + feature_idx;
                        feature_values.push_back(input[idx]);
                    }

                    means[feature_idx] = compute_mean_simd(std::span{ feature_values });
                    variances[feature_idx] = compute_variance_simd(std::span{ feature_values }, means[feature_idx]);
                });
        }

    protected:
        void normalize_simd(NormalizationData<float>& data) const noexcept override {
            const auto& config = data.config;

            // Resize buffers if needed
            if (data.mean_cache.size() != config.feature_size) {
                data.mean_cache.resize(config.feature_size);
                data.variance_cache.resize(config.feature_size);
            }

            // SIMD computation of mean/variance
            compute_statistics_simd(std::span{ data.input_data },
                std::span{ data.mean_cache },
                std::span{ data.variance_cache },
                config.batch_size,
                config.feature_size);

            // Normalize each batch using SIMD
            std::for_each(std::execution::par_unseq,
                std::views::iota(0u, config.batch_size).begin(),
                std::views::iota(0u, config.batch_size).end(),
                [&](uint32_t batch_idx) {
                    const size_t start = batch_idx * config.feature_size;
                    std::span<float> batch_output{ data.output_data.data() + start, config.feature_size };
                    const std::span<const float> batch_input{ data.input_data.data() + start, config.feature_size };

                    if (batch_input.data() != batch_output.data()) {
                        std::copy(batch_input.begin(), batch_input.end(), batch_output.begin());
                    }

                    for (uint32_t feature = 0; feature < config.feature_size; ++feature) {
                        const float mean = data.mean_cache[feature];
                        const float variance = data.variance_cache[feature];
                        const float inv_std = 1.0f / std::sqrt(variance + config.epsilon);
                        const float scale = data.scale_factors.empty() ? 1.0f : data.scale_factors[feature];
                        const float offset = data.offset_factors.empty() ? 0.0f : data.offset_factors[feature];

                        batch_output[feature] = scale * (batch_output[feature] - mean) * inv_std + offset;
                    }
                });
        }

    public:
        void normalize_scalar_fallback(NormalizationData<float>& data) const noexcept override {
            BatchNormalizationOptimized<float> fallback;
            fallback.normalize(data);
        }

        bool is_simd_supported() const noexcept override {
            return simd::SimdCapabilities::has_avx2();
        }

        std::string_view get_name() const noexcept override {
            return "BatchNormalizationSIMD";
        }

        bool requires_global_stats() const noexcept override {
            return true;
        }
    };

}
