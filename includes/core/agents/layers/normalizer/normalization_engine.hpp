#pragma once

#include "concepts.hpp"
#include "data_structures.hpp"
#include "strategy_interface.hpp"
#include "batch_normalization.hpp"
#include "layer_normalization.hpp"
#include <array>
#include <functional>
#include <memory>

namespace normalization {

    // Ultra-optimized normalization engine with strategy caching
    template<FloatingPoint T>
    class NormalizationEngine {
    private:
        using StrategyPtr = std::unique_ptr<NormalizationStrategy<T>>;
        using StrategyFactory = std::function<StrategyPtr()>;

        // Fast dispatch table for strategy creation
        std::array<StrategyFactory, constants::MAX_NORMALIZATION_TYPES> strategy_factories{};

        // Strategy cache to avoid repeated allocations
        mutable std::array<StrategyPtr, constants::MAX_NORMALIZATION_TYPES> strategy_cache{};

        // Performance statistics storage
        mutable std::array<PerformanceStats, constants::MAX_NORMALIZATION_TYPES> perf_stats{};

        // Initialization of the strategy factories
        void initialize_factories() {
            // Batch Normalization
            strategy_factories[static_cast<size_t>(NormalizationType::Batch)] =
                []() -> StrategyPtr {
                if constexpr (SimdCompatible<T>) {
                    return std::make_unique<BatchNormalizationSIMD>();
                }
                else {
                    return std::make_unique<BatchNormalizationOptimized<T>>();
                }
                };

            // Layer Normalization
            strategy_factories[static_cast<size_t>(NormalizationType::Layer)] =
                []() -> StrategyPtr {
                if constexpr (SimdCompatible<T>) {
                    return std::make_unique<LayerNormalizationSIMD>();
                }
                else {
                    return std::make_unique<LayerNormalizationOptimized<T>>();
                }
                };

            // Instance Normalization
            strategy_factories[static_cast<size_t>(NormalizationType::Instance)] =
                []() -> StrategyPtr {
                return std::make_unique<InstanceNormalization<T>>();
                };

            // Group Normalization
            strategy_factories[static_cast<size_t>(NormalizationType::Group)] =
                []() -> StrategyPtr {
                return std::make_unique<GroupNormalization<T>>();
                };
        }

        // Strategy retrieval using the cache (lazy initialization)
        NormalizationStrategy<T>* get_strategy(NormalizationType type) const {
            const auto type_idx = static_cast<size_t>(type);

            if (type_idx >= constants::MAX_NORMALIZATION_TYPES) [[unlikely]] {
                return nullptr;
            }

            // Lazy creation if not yet cached
            if (!strategy_cache[type_idx]) [[unlikely]] {
                if (strategy_factories[type_idx]) {
                    strategy_cache[type_idx] = strategy_factories[type_idx]();
                }
            }

            return strategy_cache[type_idx].get();
        }

    public:
        NormalizationEngine() {
            initialize_factories();
        }

        // Fast main normalization interface
        bool normalize(NormalizationType type, NormalizationData<T>& data) const noexcept {
            auto* strategy = get_strategy(type);
            if (!strategy) [[unlikely]] {
                return false;
            }

            // Optional input validation
            if (!strategy->validate_input(data)) [[unlikely]] {
                return false;
            }

            // Performance timing
            const auto start = std::chrono::high_resolution_clock::now();

            // Execute normalization
            strategy->normalize(data);

            // Update performance statistics
            const auto end = std::chrono::high_resolution_clock::now();
            auto& stats = perf_stats[static_cast<size_t>(type)];
            const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            stats.total_time_us += duration.count();
            stats.iterations++;
            stats.batch_size = data.config.batch_size;
            stats.feature_size = data.config.feature_size;
            stats.avg_time_per_iteration_us = stats.total_time_us / stats.iterations;

            const double elements_per_iteration = static_cast<double>(stats.batch_size * stats.feature_size);
            stats.throughput_elements_per_second = (elements_per_iteration * 1000000.0) / stats.avg_time_per_iteration_us;

            return true;
        }

        // Batch interface for processing multiple configurations
        bool normalize_batch(const std::vector<std::pair<NormalizationType, std::reference_wrapper<NormalizationData<T>>>>& batch) const noexcept {
            bool all_success = true;

            // Parallel processing using execution policy
            std::for_each(std::execution::par_unseq,
                batch.begin(), batch.end(),
                [&](const auto& item) {
                    if (!normalize(item.first, item.second.get())) {
                        all_success = false;
                    }
                });

            return all_success;
        }

        // Retrieve strategy name
        std::string_view get_strategy_name(NormalizationType type) const {
            auto* strategy = get_strategy(type);
            return strategy ? strategy->get_name() : "Unknown";
        }

        bool strategy_supports_inplace(NormalizationType type) const {
            auto* strategy = get_strategy(type);
            return strategy ? strategy->supports_inplace() : false;
        }

        bool strategy_requires_global_stats(NormalizationType type) const {
            auto* strategy = get_strategy(type);
            return strategy ? strategy->requires_global_stats() : false;
        }

        size_t estimate_complexity(NormalizationType type, const NormalizationData<T>& data) const {
            auto* strategy = get_strategy(type);
            return strategy ? strategy->estimate_complexity(data) : 0;
        }

        // Access to performance statistics
        const PerformanceStats& get_performance_stats(NormalizationType type) const {
            return perf_stats[static_cast<size_t>(type)];
        }

        void reset_performance_stats() const {
            for (auto& stats : perf_stats) {
                stats = PerformanceStats{};
            }
        }

        // Dynamic registration of custom strategies
        template<typename StrategyType, typename... Args>
        void register_custom_strategy(NormalizationType type, Args&&... args) {
            static_assert(std::is_base_of_v<NormalizationStrategy<T>, StrategyType>,
                "StrategyType must inherit from NormalizationStrategy");

            const auto type_idx = static_cast<size_t>(type);
            if (type_idx < constants::MAX_NORMALIZATION_TYPES) {
                strategy_factories[type_idx] = [args...]() -> StrategyPtr {
                    return std::make_unique<StrategyType>(args...);
                    };

                // Invalidate the cached strategy
                strategy_cache[type_idx].reset();
            }
        }

        // Preload strategies (useful to avoid runtime allocations)
        void preload_strategies(const std::vector<NormalizationType>& types) {
            for (auto type : types) {
                get_strategy(type);  // Force creation
            }
        }

        void preload_all_strategies() {
            preload_strategies({
                NormalizationType::Batch,
                NormalizationType::Layer,
                NormalizationType::Instance,
                NormalizationType::Group
                });
        }

        // Debug utility to print performance summary
        void print_performance_summary() const {
            std::cout << "\n=== Normalization Performance Summary ===\n";

            const std::array<std::string_view, 4> type_names = {
                "Batch", "Layer", "Instance", "Group"
            };

            for (size_t i = 0; i < 4; ++i) {
                const auto& stats = perf_stats[i];
                if (stats.iterations > 0) {
                    std::cout << type_names[i] << " Normalization:\n"
                        << "  Strategy: " << get_strategy_name(static_cast<NormalizationType>(i)) << "\n"
                        << "  Iterations: " << stats.iterations << "\n"
                        << "  Avg Time: " << stats.avg_time_per_iteration_us << " μs\n"
                        << "  Throughput: " << stats.throughput_elements_per_second << " elements/sec\n"
                        << "  Last Batch Size: " << stats.batch_size << "x" << stats.feature_size << "\n\n";
                }
            }
        }

        // Auto-tuning: automatically selects the best strategy based on heuristics
        NormalizationType auto_select_best_strategy(const NormalizationData<T>& data) const {
            // Heuristics based on data size and shape
            const size_t total_elements = static_cast<size_t>(data.config.batch_size) * data.config.feature_size;

            if (data.config.batch_size > 1 && total_elements > 10000) {
                // Large batch size -> prefer Batch Normalization
                return NormalizationType::Batch;
            }
            else if (data.config.feature_size > 512) {
                // Long sequences -> Layer Normalization with SIMD
                return NormalizationType::Layer;
            }
            else if (data.config.batch_size == 1) {
                // Inference case -> Instance Normalization
                return NormalizationType::Instance;
            }
            else {
                // Default fallback -> Layer Normalization
                return NormalizationType::Layer;
            }
        }

        // Normalize using automatic strategy selection
        bool normalize_auto(NormalizationData<T>& data) const noexcept {
            const auto best_type = auto_select_best_strategy(data);
            return normalize(best_type, data);
        }
    };

    // Factory function to create and initialize a normalization engine
    template<FloatingPoint T>
    std::unique_ptr<NormalizationEngine<T>> create_normalization_engine() {
        auto engine = std::make_unique<NormalizationEngine<T>>();
        engine->preload_all_strategies();  // Preload for better runtime performance
        return engine;
    }

    // Aliases for most common types
    using FloatNormalizationEngine = NormalizationEngine<float>;
    using DoubleNormalizationEngine = NormalizationEngine<double>;

} // namespace deep_qn::normalization::v2
