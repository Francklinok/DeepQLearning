#pragma once

#include "concepts.hpp"
#include "data_structures.hpp"
#include <string_view>

namespace normalization {

    // Base interface for all normalization strategies
    template<FloatingPoint T>
    class NormalizationStrategy {
    public:
        virtual ~NormalizationStrategy() = default;

        // Main normalization method (must be thread-safe)
        virtual void normalize(NormalizationData<T>& data) const noexcept = 0;

        // Strategy name (for logging, debugging, profiling)
        virtual std::string_view get_name() const noexcept = 0;

        // Optional validation hook to check if input data is suitable
        virtual bool validate_input(const NormalizationData<T>& data) const noexcept {
            return data.is_valid() && data.config.epsilon > T{ 0 };
        }

        // Estimate the computational complexity (useful for scheduling/prioritizing)
        virtual size_t estimate_complexity(const NormalizationData<T>& data) const noexcept {
            return static_cast<size_t>(data.config.batch_size) * data.config.feature_size;
        }

        // Indicates whether the strategy supports in-place processing
        virtual bool supports_inplace() const noexcept { return true; }

        // Indicates whether global statistics (mean, variance) are required
        virtual bool requires_global_stats() const noexcept { return false; }
    };

    // Specialized interface for SIMD-capable normalization strategies
    template<SimdCompatible T>
    class SimdNormalizationStrategy : public NormalizationStrategy<T> {
    public:
        // Check if SIMD is supported on the current hardware/runtime
        virtual bool is_simd_supported() const noexcept = 0;

        // Fallback method when SIMD is not available (scalar implementation)
        virtual void normalize_scalar_fallback(NormalizationData<T>& data) const noexcept = 0;

        // Final normalize method that selects SIMD or scalar path
        void normalize(NormalizationData<T>& data) const noexcept override final {
            if (is_simd_supported()) {
                normalize_simd(data);
            }
            else {
                normalize_scalar_fallback(data);
            }
        }

    protected:
        // Actual SIMD implementation (must be overridden by derived classes)
        virtual void normalize_simd(NormalizationData<T>& data) const noexcept = 0;
    };

} 
