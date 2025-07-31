#pragma once

/**
 * @file normalization_system.hpp
 * @brief Ultra-Optimized Normalization System for Deep Learning
 *
 * This system implements cutting-edge C++ optimization techniques for deep learning
 * normalization operations, featuring:
 * - Data-Oriented Design (DOD) for cache efficiency
 * - SIMD vectorization (AVX2) for parallel computation
 * - Modern parallelism via std::execution
 * - Modular strategy-based architecture
 * - Native TensorFlow integration (optional)
 *
 * @author Deep QN Team
 * @version 1.0
 */

#include <cstdint>
#include <memory>
#include <string_view>

#include "concepts.hpp"
#include "data_structures.hpp"
#include "strategy_interface.hpp"
#include "simd_utils.hpp"
#include "batch_normalization.hpp"
#include "layer_normalization.hpp"
#include "normalization_engine.hpp"
#include "performance_profiler.hpp"

#ifdef TENSORFLOW_ENABLED
#include "tensorflow_adapter.hpp"
#endif

 
namespace normalization {

    constexpr std::string_view SYSTEM_VERSION = "2.0.0";

    struct BuildInfo {
        static constexpr std::string_view version = SYSTEM_VERSION;

        static constexpr std::string_view compiler =
#ifdef __clang__
            "Clang";
#elif defined(__GNUC__)
            "GCC";
#elif defined(_MSC_VER)
            "MSVC";
#else
            "Unknown";
#endif

        static constexpr bool simd_enabled =
#ifdef __AVX2__
            true;
#else
            false;
#endif

        static constexpr bool tensorflow_enabled =
#ifdef TENSORFLOW_ENABLED
            true;
#else
            false;
#endif

        static constexpr bool debug_build =
#ifdef NDEBUG
            false;
#else
            true;
#endif
    };

    /**
     * @brief High-level interface for quick usage.
     */
    template<FloatingPoint T = float>
    class QuickNormalizer {
    private:
        std::unique_ptr<NormalizationEngine<T>> engine;
        NormalizationData<T> data_buffer;

    public:
        QuickNormalizer() : engine(create_normalization_engine<T>()) {}

        /**
         * @brief Perform fast batch normalization.
         */
        bool batch_normalize(const T* input, T* output,
            uint32_t batch_size, uint32_t feature_size,
            T epsilon = T{ 1e-5 }) {
            data_buffer.resize_for_batch(batch_size, feature_size);
            data_buffer.config.epsilon = epsilon;

            std::copy(input, input + batch_size * feature_size, data_buffer.input_data.begin());

            bool success = engine->normalize(NormalizationType::Batch, data_buffer);

            if (success) {
                std::copy(data_buffer.output_data.begin(), data_buffer.output_data.end(), output);
            }

            return success;
        }

        /**
         * @brief Perform fast layer normalization.
         */
        bool layer_normalize(const T* input, T* output,
            uint32_t batch_size, uint32_t sequence_length,
            T epsilon = T{ 1e-5 }) {
            data_buffer.resize_for_batch(batch_size, sequence_length);
            data_buffer.config.epsilon = epsilon;

            std::copy(input, input + batch_size * sequence_length, data_buffer.input_data.begin());

            bool success = engine->normalize(NormalizationType::Layer, data_buffer);

            if (success) {
                std::copy(data_buffer.output_data.begin(), data_buffer.output_data.end(), output);
            }

            return success;
        }

        /**
         * @brief Normalize using affine scaling and offset factors.
         */
        bool normalize_with_affine(NormalizationType type,
            const T* input, T* output,
            const T* scale, const T* offset,
            uint32_t batch_size, uint32_t feature_size,
            T epsilon = T{ 1e-5 }) {
            data_buffer.resize_for_batch(batch_size, feature_size);
            data_buffer.config.epsilon = epsilon;

            std::copy(input, input + batch_size * feature_size, data_buffer.input_data.begin());

            if (scale) {
                data_buffer.scale_factors.assign(scale, scale + feature_size);
            }
            if (offset) {
                data_buffer.offset_factors.assign(offset, offset + feature_size);
            }

            bool success = engine->normalize(type, data_buffer);

            if (success) {
                std::copy(data_buffer.output_data.begin(), data_buffer.output_data.end(), output);
            }

            return success;
        }

        NormalizationEngine<T>& get_engine() { return *engine; }
        const NormalizationEngine<T>& get_engine() const { return *engine; }
    };

    /**
     * @brief Utility functions for diagnostics, testing, and benchmarking.
     */
    namespace utils {

        inline void print_system_info() {
            std::cout << "=== Deep QN Normalization System v" << BuildInfo::version << " ===\n";
            std::cout << "Compiler: " << BuildInfo::compiler << "\n";
            std::cout << "SIMD Support: " << (BuildInfo::simd_enabled ? "? AVX2" : "? No AVX2") << "\n";
            std::cout << "TensorFlow: " << (BuildInfo::tensorflow_enabled ? "? Enabled" : "? Disabled") << "\n";
            std::cout << "Build Type: " << (BuildInfo::debug_build ? "Debug" : "Release") << "\n";

            if constexpr (BuildInfo::simd_enabled) {
                std::cout << "Runtime SIMD: " << (simd::SimdCapabilities::has_avx2() ? "? AVX2" : "? No AVX2") << "\n";
                std::cout << "Runtime FMA: " << (simd::SimdCapabilities::has_fma() ? "? FMA" : "? No FMA") << "\n";
            }

            std::cout << std::string(60, '=') << "\n\n";
        }

        template<FloatingPoint T = float>
        bool quick_test() {
            std::cout << "Running quick system test...\n";

            QuickNormalizer<T> normalizer;

            constexpr uint32_t batch_size = 32;
            constexpr uint32_t feature_size = 128;
            const size_t total_size = batch_size * feature_size;

            std::vector<T> input(total_size);
            std::vector<T> output(total_size);

            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<T> dist(T{ 0 }, T{ 1 });
            std::generate(input.begin(), input.end(), [&]() { return dist(gen); });

            bool batch_success = normalizer.batch_normalize(input.data(), output.data(), batch_size, feature_size);
            bool layer_success = normalizer.layer_normalize(input.data(), output.data(), batch_size, feature_size);

            bool output_valid = std::all_of(output.begin(), output.end(),
                [](T val) { return std::isfinite(val); });

            std::cout << "Batch Normalization: " << (batch_success ? "? PASS" : "? FAIL") << "\n";
            std::cout << "Layer Normalization: " << (layer_success ? "? PASS" : "? FAIL") << "\n";
            std::cout << "Output Validation: " << (output_valid ? "? PASS" : "? FAIL") << "\n";

            bool overall_success = batch_success && layer_success && output_valid;
            std::cout << "Overall Test: " << (overall_success ? "? PASS" : "? FAIL") << "\n\n";

            return overall_success;
        }

        template<FloatingPoint T = float>
        void run_quick_benchmark() {
            std::cout << "Running quick performance benchmark...\n\n";

            PerformanceProfiler<T> profiler;

            const std::vector<std::pair<size_t, size_t>> test_sizes = {
                {32, 256},   // Small
                {128, 512},  // Medium
                {256, 1024}  // Large
            };

            for (const auto& [batch, features] : test_sizes) {
                std::cout << "=== Size: " << batch << "x" << features << " ===\n";
                profiler.benchmark_all_strategies(batch, features, 100);
                std::cout << "\n";
            }
        }

        struct UsageRecommendation {
            NormalizationType recommended_type;
            std::string reason;
            std::vector<std::string> optimizations;
        };

        inline UsageRecommendation get_usage_recommendation(
            uint32_t batch_size,
            uint32_t feature_size,
            bool is_training = true,
            bool requires_stability = false) {

            UsageRecommendation rec;

            const size_t total_elements = static_cast<size_t>(batch_size) * feature_size;

            if (is_training && batch_size > 1 && total_elements > 1000) {
                rec.recommended_type = NormalizationType::Batch;
                rec.reason = "Training with large batches benefits from batch statistics.";
                rec.optimizations = { "Use SIMD", "Enable global statistics caching" };
            }
            else if (feature_size > 256 && batch_size <= 64) {
                rec.recommended_type = NormalizationType::Layer;
                rec.reason = "Long sequences benefit from layer normalization.";
                rec.optimizations = { "Use Welford algorithm", "Enable SIMD" };
            }
            else if (batch_size == 1 || !is_training) {
                rec.recommended_type = NormalizationType::Instance;
                rec.reason = "Ideal for single instance or inference.";
                rec.optimizations = { "Minimize memory allocations", "Use in-place computation" };
            }
            else {
                rec.recommended_type = NormalizationType::Layer;
                rec.reason = "Layer normalization is the most versatile for general use.";
                rec.optimizations = { "Standard optimizations apply" };
            }

            if (requires_stability) {
                rec.optimizations.push_back("Use higher epsilon (1e-6 to 1e-4)");
                rec.optimizations.push_back("Consider double precision for critical tasks");
            }

            return rec;
        }

        inline void print_recommendation(const UsageRecommendation& rec) {
            std::cout << "=== Usage Recommendation ===\n";
            std::cout << "Recommended Type: " << static_cast<int>(rec.recommended_type) << "\n";
            std::cout << "Reason: " << rec.reason << "\n";
            std::cout << "Optimizations:\n";
            for (const auto& opt : rec.optimizations) {
                std::cout << "  - " << opt << "\n";
            }
            std::cout << "\n";
        }
    }

    // Type aliases
    using FastNormalizer = QuickNormalizer<float>;
    using PreciseNormalizer = QuickNormalizer<double>;

    // Factory functions
    template<FloatingPoint T = float>
    std::unique_ptr<QuickNormalizer<T>> create_quick_normalizer() {
        return std::make_unique<QuickNormalizer<T>>();
    }

    inline std::unique_ptr<FastNormalizer> create_fast_normalizer() {
        return create_quick_normalizer<float>();
    }

    inline std::unique_ptr<PreciseNormalizer> create_precise_normalizer() {
        return create_quick_normalizer<double>();
    }

} 

// Version macros
#define DEEP_QN_NORM_VERSION_MAJOR 2
#define DEEP_QN_NORM_VERSION_MINOR 0
#define DEEP_QN_NORM_VERSION_PATCH 0

#define DEEP_QN_NORM_STRINGIFY(x) #x
#define DEEP_QN_NORM_VERSION_STRING \
    DEEP_QN_NORM_STRINGIFY(DEEP_QN_NORM_VERSION_MAJOR) "." \
    DEEP_QN_NORM_STRINGIFY(DEEP_QN_NORM_VERSION_MINOR) "." \
    DEEP_QN_NORM_STRINGIFY(DEEP_QN_NORM_VERSION_PATCH)

/**
 * @brief Example usage
 * @code
 * #include "normalization_system.hpp"
 *
 * int main() {
 *     using namespace normalization;
 *     utils::print_system_info();
 *
 *     if (!utils::quick_test<float>()) return -1;
 *
 *     auto normalizer = create_fast_normalizer();
 *
 *     std::vector<float> input(64 * 256);
 *     std::vector<float> output(input.size());
 *
 *     // Fill input with values...
 *     bool success = normalizer->batch_normalize(input.data(), output.data(), 64, 256);
 *
 *     if (success) {
 *         // Use normalized output
 *     }
 *     return 0;
 * }
 * @endcode
 */
