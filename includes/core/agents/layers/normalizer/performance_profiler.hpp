#pragma once

#include "normalization_engine.hpp"
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <map>

namespace normalization {

    // Profileur de performances avancé
    template<FloatingPoint T>
    class PerformanceProfiler {
    private:
        std::unique_ptr<NormalizationEngine<T>> engine;
        std::mt19937 rng;
        std::normal_distribution<T> dist;

        // Résultats de benchmarks
        struct BenchmarkResult {
            std::string strategy_name;
            NormalizationType type;
            size_t batch_size;
            size_t feature_size;
            size_t iterations;
            double total_time_us;
            double avg_time_us;
            double min_time_us;
            double max_time_us;
            double std_dev_us;
            double throughput_elements_per_sec;
            double memory_bandwidth_gb_per_sec;

            BenchmarkResult() : total_time_us(0), avg_time_us(0), min_time_us(0),
                max_time_us(0), std_dev_us(0), throughput_elements_per_sec(0),
                memory_bandwidth_gb_per_sec(0) {
            }
        };

        std::vector<BenchmarkResult> benchmark_history;

    public:
        explicit PerformanceProfiler(uint64_t seed = std::random_device{}())
            : engine(create_normalization_engine<T>()), rng(seed), dist(T{ 0 }, T{ 1 }) {
        }

        // Benchmark principal d'une stratégie
        BenchmarkResult benchmark_strategy(NormalizationType type,
            size_t batch_size,
            size_t feature_size,
            size_t iterations = 1000,
            bool verbose = true) {
            BenchmarkResult result;
            result.type = type;
            result.batch_size = batch_size;
            result.feature_size = feature_size;
            result.iterations = iterations;
            result.strategy_name = std::string(engine->get_strategy_name(type));

            // Préparation des données
            NormalizationData<T> data;
            setup_test_data(data, batch_size, feature_size);

            // Warmup pour stabiliser les performances
            for (size_t i = 0; i < 10; ++i) {
                engine->normalize(type, data);
            }

            // Mesures de performance
            std::vector<double> iteration_times;
            iteration_times.reserve(iterations);

            const auto total_start = std::chrono::high_resolution_clock::now();

            for (size_t i = 0; i < iterations; ++i) {
                const auto start = std::chrono::high_resolution_clock::now();

                bool success = engine->normalize(type, data);
                if (!success) {
                    std::cerr << "Normalization failed at iteration " << i << std::endl;
                    break;
                }

                const auto end = std::chrono::high_resolution_clock::now();
                const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
                iteration_times.push_back(duration.count() / 1000.0);  // Convert to microseconds
            }

            const auto total_end = std::chrono::high_resolution_clock::now();
            const auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);

            // Calcul des statistiques
            result.total_time_us = total_duration.count();
            result.avg_time_us = result.total_time_us / iterations;

            if (!iteration_times.empty()) {
                result.min_time_us = *std::min_element(iteration_times.begin(), iteration_times.end());
                result.max_time_us = *std::max_element(iteration_times.begin(), iteration_times.end());

                // Calcul de l'écart-type
                double variance = 0.0;
                for (double time : iteration_times) {
                    variance += (time - result.avg_time_us) * (time - result.avg_time_us);
                }
                result.std_dev_us = std::sqrt(variance / iterations);
            }

            // Métriques de performance
            const double elements_per_iteration = static_cast<double>(batch_size * feature_size);
            result.throughput_elements_per_sec = (elements_per_iteration * 1000000.0) / result.avg_time_us;

            // Estimation de la bande passante mémoire (lecture + écriture)
            const double bytes_per_iteration = elements_per_iteration * sizeof(T) * 2;  // Input + Output
            result.memory_bandwidth_gb_per_sec = (bytes_per_iteration * 1000000.0) / (result.avg_time_us * 1e9);

            benchmark_history.push_back(result);

            if (verbose) {
                print_benchmark_result(result);
            }

            return result;
        }

        // Benchmark comparatif de toutes les stratégies
        void benchmark_all_strategies(size_t batch_size,
            size_t feature_size,
            size_t iterations = 1000) {
            std::cout << "\n" << std::string(80, '=') << "\n";
            std::cout << "COMPREHENSIVE NORMALIZATION BENCHMARK\n";
            std::cout << "Batch Size: " << batch_size << ", Feature Size: " << feature_size << "\n";
            std::cout << "Iterations: " << iterations << "\n";
            std::cout << std::string(80, '=') << "\n\n";

            const std::vector<NormalizationType> types = {
                NormalizationType::Batch,
                NormalizationType::Layer,
                NormalizationType::Instance,
                NormalizationType::Group
            };

            std::vector<BenchmarkResult> results;

            for (auto type : types) {
                std::cout << "Benchmarking " << engine->get_strategy_name(type) << "...\n";
                auto result = benchmark_strategy(type, batch_size, feature_size, iterations, false);
                results.push_back(result);
                std::cout << "✓ Completed\n\n";
            }

            // Comparaison des résultats
            std::cout << "\n" << std::string(80, '-') << "\n";
            std::cout << "PERFORMANCE COMPARISON\n";
            std::cout << std::string(80, '-') << "\n";

            // Tri par performance (temps moyen)
            std::sort(results.begin(), results.end(),
                [](const BenchmarkResult& a, const BenchmarkResult& b) {
                    return a.avg_time_us < b.avg_time_us;
                });

            std::cout << std::left << std::setw(20) << "Strategy"
                << std::setw(12) << "Avg Time(μs)"
                << std::setw(15) << "Throughput(M/s)"
                << std::setw(15) << "Bandwidth(GB/s)"
                << std::setw(10) << "Speedup" << "\n";
            std::cout << std::string(80, '-') << "\n";

            const double baseline_time = results[0].avg_time_us;

            for (const auto& result : results) {
                const double speedup = baseline_time / result.avg_time_us;
                const double throughput_millions = result.throughput_elements_per_sec / 1e6;

                std::cout << std::left << std::setw(20) << result.strategy_name
                    << std::setw(12) << std::fixed << std::setprecision(2) << result.avg_time_us
                    << std::setw(15) << std::fixed << std::setprecision(1) << throughput_millions
                    << std::setw(15) << std::fixed << std::setprecision(2) << result.memory_bandwidth_gb_per_sec
                    << std::setw(10) << std::fixed << std::setprecision(2) << speedup << "x\n";
            }
        }

        // Benchmark de scaling (différentes tailles)
        void benchmark_scaling(NormalizationType type,
            const std::vector<std::pair<size_t, size_t>>& sizes,
            size_t iterations = 500) {
            std::cout << "\n" << std::string(80, '=') << "\n";
            std::cout << "SCALING BENCHMARK - " << engine->get_strategy_name(type) << "\n";
            std::cout << std::string(80, '=') << "\n\n";

            std::cout << std::left << std::setw(10) << "Batch"
                << std::setw(10) << "Features"
                << std::setw(12) << "Elements"
                << std::setw(12) << "Time(μs)"
                << std::setw(15) << "Throughput(M/s)"
                << std::setw(15) << "Efficiency" << "\n";
            std::cout << std::string(80, '-') << "\n";

            BenchmarkResult baseline;
            bool first = true;

            for (const auto& [batch_size, feature_size] : sizes) {
                auto result = benchmark_strategy(type, batch_size, feature_size, iterations, false);

                if (first) {
                    baseline = result;
                    first = false;
                }

                const double elements = static_cast<double>(batch_size * feature_size);
                const double throughput_millions = result.throughput_elements_per_sec / 1e6;
                const double efficiency = (baseline.throughput_elements_per_sec / result.throughput_elements_per_sec) *
                    (elements / (baseline.batch_size * baseline.feature_size));

                std::cout << std::left << std::setw(10) << batch_size
                    << std::setw(10) << feature_size
                    << std::setw(12) << static_cast<size_t>(elements)
                    << std::setw(12) << std::fixed << std::setprecision(2) << result.avg_time_us
                    << std::setw(15) << std::fixed << std::setprecision(1) << throughput_millions
                    << std::setw(15) << std::fixed << std::setprecision(3) << efficiency << "\n";
            }
        }

        // Génération de rapport détaillé
        void generate_report(const std::string& filename = "normalization_benchmark_report.txt") const {
            std::ofstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Cannot open file: " << filename << std::endl;
                return;
            }

            file << "DEEP QN NORMALIZATION SYSTEM - PERFORMANCE REPORT\n";
            file << "Generated: " << std::chrono::system_clock::now().time_since_epoch().count() << "\n";
            file << std::string(80, '=') << "\n\n";

            // Informations système
            file << "SYSTEM INFORMATION:\n";
            file << "- Data Type: " << (sizeof(T) == 4 ? "float" : "double") << " (" << sizeof(T) << " bytes)\n";
            file << "- SIMD Support: " << (simd::SimdCapabilities::has_avx2() ? "AVX2 ✓" : "No AVX2") << "\n";
            file << "- FMA Support: " << (simd::SimdCapabilities::has_fma() ? "FMA ✓" : "No FMA") << "\n\n";

            // Résumé des benchmarks
            std::map<NormalizationType, std::vector<BenchmarkResult>> grouped_results;
            for (const auto& result : benchmark_history) {
                grouped_results[result.type].push_back(result);
            }

            for (const auto& [type, results] : grouped_results) {
                file << "STRATEGY: " << results[0].strategy_name << "\n";
                file << std::string(50, '-') << "\n";

                for (const auto& result : results) {
                    file << "Configuration: " << result.batch_size << "x" << result.feature_size
                        << " (" << result.iterations << " iterations)\n";
                    file << "  Average Time: " << std::fixed << std::setprecision(2) << result.avg_time_us << " μs\n";
                    file << "  Min/Max Time: " << result.min_time_us << "/" << result.max_time_us << " μs\n";
                    file << "  Std Deviation: " << result.std_dev_us << " μs\n";
                    file << "  Throughput: " << std::fixed << std::setprecision(1)
                        << (result.throughput_elements_per_sec / 1e6) << " M elements/sec\n";
                    file << "  Memory Bandwidth: " << std::fixed << std::setprecision(2)
                        << result.memory_bandwidth_gb_per_sec << " GB/sec\n\n";
                }
            }

            file.close();
            std::cout << "Report saved to: " << filename << std::endl;
        }

        // Nettoyage de l'historique
        void clear_history() {
            benchmark_history.clear();
            engine->reset_performance_stats();
        }

    private:
        void setup_test_data(NormalizationData<T>& data, size_t batch_size, size_t feature_size) {
            data.resize_for_batch(static_cast<uint32_t>(batch_size), static_cast<uint32_t>(feature_size));

            // Génération de données réalistes
            std::generate(data.input_data.begin(), data.input_data.end(),
                [this]() { return dist(rng); });

            // Initialisation des facteurs d'échelle (optionnel)
            if (rng() % 2 == 0) {  // 50% de chance d'avoir des facteurs
                data.scale_factors.resize(feature_size);
                data.offset_factors.resize(feature_size);

                std::generate(data.scale_factors.begin(), data.scale_factors.end(),
                    [this]() { return T{ 0.5 } + dist(rng) * T { 0.5 }; });  // [0.5, 1.5]

                std::generate(data.offset_factors.begin(), data.offset_factors.end(),
                    [this]() { return dist(rng) * T { 0.1 }; });  // [-0.1, 0.1]
            }
        }

        void print_benchmark_result(const BenchmarkResult& result) const {
            std::cout << "Strategy: " << result.strategy_name << "\n";
            std::cout << "Configuration: " << result.batch_size << "x" << result.feature_size
                << " (" << result.iterations << " iterations)\n";
            std::cout << "Average Time: " << std::fixed << std::setprecision(2) << result.avg_time_us << " μs\n";
            std::cout << "Min/Max Time: " << result.min_time_us << "/" << result.max_time_us << " μs\n";
            std::cout << "Throughput: " << std::fixed << std::setprecision(1)
                << (result.throughput_elements_per_sec / 1e6) << " M elements/sec\n";
            std::cout << "Memory Bandwidth: " << std::fixed << std::setprecision(2)
                << result.memory_bandwidth_gb_per_sec << " GB/sec\n\n";
        }
    };

    // Fonctions utilitaires pour benchmarks rapides
    template<FloatingPoint T>
    void quick_benchmark() {
        PerformanceProfiler<T> profiler;

        // Benchmark rapide avec différentes tailles
        const std::vector<std::pair<size_t, size_t>> sizes = {
            {32, 256},    // Petit
            {64, 512},    // Moyen
            {128, 1024},  // Grand
            {256, 2048}   // Très grand
        };

        for (const auto& [batch, features] : sizes) {
            std::cout << "\n--- Benchmark " << batch << "x" << features << " ---\n";
            profiler.benchmark_all_strategies(batch, features, 100);
        }

        profiler.generate_report();
    }

    // Spécialisations pour les types courants
    using FloatProfiler = PerformanceProfiler<float>;
    using DoubleProfiler = PerformanceProfiler<double>;

} 