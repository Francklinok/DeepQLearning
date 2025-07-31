#pragma once

#include "strategy_interface.hpp"
#include "simd_utils.hpp"
#include <execution>
#include <algorithm>
#include <ranges>

namespace normalization {

    // Layer Normalization optimized using Welford's algorithm
    template<FloatingPoint T>
    class LayerNormalizationOptimized final : public NormalizationStrategy<T> {
    private:
        // Welford's algorithm for numerically stable mean and variance
        struct WelfordStats {
            T mean;
            T m2;  // Sum of squares of differences
            size_t count;

            WelfordStats() : mean(T{ 0 }), m2(T{ 0 }), count(0) {}

            void update(T value) noexcept {
                ++count;
                const T delta = value - mean;
                mean += delta / static_cast<T>(count);
                const T delta2 = value - mean;
                m2 += delta * delta2;
            }

            T get_variance() const noexcept {
                return count > 1 ? m2 / static_cast<T>(count) : T{ 0 };
            }
        };

        // Normalize a single sequence using Welford's algorithm
        void normalize_sequence_welford(std::span<T> sequence, T epsilon) const noexcept {
            if (sequence.empty()) return;

            WelfordStats stats;

            // Incremental computation of mean and variance
            for (const T value : sequence) {
                stats.update(value);
            }

            const T variance = stats.get_variance();
            const T inv_std = T{ 1 } / std::sqrt(variance + epsilon);

            // In-place normalization
            std::transform(std::execution::par_unseq,
                sequence.begin(), sequence.end(), sequence.begin(),
                [mean = stats.mean, inv_std](T value) noexcept -> T {
                    return (value - mean) * inv_std;
                });
        }

    public:
        void normalize(NormalizationData<T>& data) const noexcept override {
            const auto& config = data.config;

            if (!validate_input(data)) return;

            // Parallel batch processing
            std::for_each(std::execution::par_unseq,
                std::views::iota(0u, config.batch_size).begin(),
                std::views::iota(0u, config.batch_size).end(),
                [&](uint32_t batch_idx) {
                    const size_t start = batch_idx * config.feature_size;

                    std::span<const T> input_sequence{
                        data.input_data.data() + start, config.feature_size
                    };
                    std::span<T> output_sequence{
                        data.output_data.data() + start, config.feature_size
                    };

                    // Optimized copy if needed
                    if (input_sequence.data() != output_sequence.data()) {
                        std::copy(std::execution::par_unseq,
                            input_sequence.begin(), input_sequence.end(),
                            output_sequence.begin());
                    }

                    // Normalize the sequence
                    normalize_sequence_welford(output_sequence, config.epsilon);

                    // Apply scaling and offset if present
                    if (!data.scale_factors.empty() || !data.offset_factors.empty()) {
                        for (size_t i = 0; i < output_sequence.size(); ++i) {
                            const T scale = data.scale_factors.empty() ? T{ 1 } : data.scale_factors[i];
                            const T offset = data.offset_factors.empty() ? T{ 0 } : data.offset_factors[i];
                            output_sequence[i] = scale * output_sequence[i] + offset;
                        }
                    }
                });
        }

        std::string_view get_name() const noexcept override {
            return "LayerNormalizationOptimized";
        }

        bool requires_global_stats() const noexcept override {
            return false;  // Layer Norm computes stats per sequence
        }
    };

    // SIMD-accelerated Layer Normalization for float
    template<>
    class LayerNormalizationSIMD final : public SimdNormalizationStrategy<float> {
    private:
        // SIMD normalization of a single sequence
        void normalize_sequence_simd(std::span<float> sequence, float epsilon) const noexcept {
            if (sequence.size() < simd::f32::VECTOR_SIZE) {
                // Fallback for small sequences
                LayerNormalizationOptimized<float> fallback;
                fallback.normalize_sequence_welford(sequence, epsilon);
                return;
            }

            using namespace simd::f32;

            const float mean = compute_mean_simd(sequence);
            const float variance = compute_variance_simd(sequence, mean);
            const float inv_std = 1.0f / std::sqrt(variance + epsilon);

            normalize_inplace_simd(sequence, mean, inv_std);
        }

    protected:
        void normalize_simd(NormalizationData<float>& data) const noexcept override {
            const auto& config = data.config;

            // Parallel batch processing with SIMD
            std::for_each(std::execution::par_unseq,
                std::views::iota(0u, config.batch_size).begin(),
                std::views::iota(0u, config.batch_size).end(),
                [&](uint32_t batch_idx) {
                    const size_t start = batch_idx * config.feature_size;

                    std::span<const float> input_sequence{
                        data.input_data.data() + start, config.feature_size
                    };
                    std::span<float> output_sequence{
                        data.output_data.data() + start, config.feature_size
                    };

                    if (input_sequence.data() != output_sequence.data()) {
                        std::copy(std::execution::par_unseq,
                            input_sequence.begin(), input_sequence.end(),
                            output_sequence.begin());
                    }

                    normalize_sequence_simd(output_sequence, config.epsilon);

                    // Apply scale and offset with SIMD if possible
                    if (!data.scale_factors.empty() || !data.offset_factors.empty()) {
                        for (size_t i = 0; i < output_sequence.size(); ++i) {
                            const float scale = data.scale_factors.empty() ? 1.0f : data.scale_factors[i];
                            const float offset = data.offset_factors.empty() ? 0.0f : data.offset_factors[i];

                            if (scale != 1.0f || offset != 0.0f) {
                                std::span<float> block{ output_sequence.data() + i,
                                                        std::min(simd::f32::VECTOR_SIZE, output_sequence.size() - i) };

                                if (block.size() >= simd::f32::VECTOR_SIZE) {
                                    simd::f32::normalize_with_affine_simd(block, 0.0f, 1.0f, scale, offset);
                                    i += simd::f32::VECTOR_SIZE - 1;  // -1 to account for loop increment
                                }
                                else {
                                    output_sequence[i] = scale * output_sequence[i] + offset;
                                }
                            }
                        }
                    }
                });
        }

    public:
        void normalize_scalar_fallback(NormalizationData<float>& data) const noexcept override {
            LayerNormalizationOptimized<float> fallback;
            fallback.normalize(data);
        }

        bool is_simd_supported() const noexcept override {
            return simd::SimdCapabilities::has_avx2();
        }

        std::string_view get_name() const noexcept override {
            return "LayerNormalizationSIMD";
        }

        bool requires_global_stats() const noexcept override {
            return false;
        }
    };

    // Instance Normalization (one normalization per instance/channel)
    template<FloatingPoint T>
    class InstanceNormalization final : public NormalizationStrategy<T> {
    public:
        void normalize(NormalizationData<T>& data) const noexcept override {
            // Instance Norm is similar to Layer Norm, applied per channel
            LayerNormalizationOptimized<T> layer_norm;
            layer_norm.normalize(data);
        }

        std::string_view get_name() const noexcept override {
            return "InstanceNormalization";
        }
    };

    // Group Normalization (normalize over groups of channels)
    template<FloatingPoint T>
    class GroupNormalization final : public NormalizationStrategy<T> {
    private:
        uint32_t num_groups;

    public:
        explicit GroupNormalization(uint32_t groups = 32) : num_groups(groups) {}

        void normalize(NormalizationData<T>& data) const noexcept override {
            const auto& config = data.config;

            if (config.feature_size % num_groups != 0) {
                // Fallback to Layer Norm if group division is not clean
                LayerNormalizationOptimized<T> fallback;
                fallback.normalize(data);
                return;
            }

            const uint32_t group_size = config.feature_size / num_groups;

            // Parallel processing by batch and group
            std::for_each(std::execution::par_unseq,
                std::views::iota(0u, config.batch_size * num_groups).begin(),
                std::views::iota(0u, config.batch_size * num_groups).end(),
                [&](uint32_t idx) {
                    const uint32_t batch_idx = idx / num_groups;
                    const uint32_t group_idx = idx % num_groups;

                    const size_t start = batch_idx * config.feature_size + group_idx * group_size;

                    std::span<const T> input_group{
                        data.input_data.data() + start, group_size
                    };
                    std::span<T> output_group{
                        data.output_data.data() + start, group_size
                    };

                    if (input_group.data() != output_group.data()) {
                        std::copy(input_group.begin(), input_group.end(), output_group.begin());
                    }

                    // Normalize the group (like Layer Norm)
                    LayerNormalizationOptimized<T> normalizer;
                    normalizer.normalize_sequence_welford(output_group, config.epsilon);
                });
        }

        std::string_view get_name() const noexcept override {
            return "GroupNormalization";
        }

        uint32_t get_num_groups() const noexcept { return num_groups; }
        void set_num_groups(uint32_t groups) noexcept { num_groups = groups; }
    };

} 
