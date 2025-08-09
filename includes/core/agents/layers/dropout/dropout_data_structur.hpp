#pragma once // Ensures this header is included only once during compilation

// Required standard headers
#include <vector>         // For std::vector
#include <random>         // For std::mt19937, std::random_device
#include <thread>         // For std::thread::hardware_concurrency
#include <atomic>         // For std::atomic
#include <cstddef>        // For size_t
#include <concepts>       // For Numeric concept (if defined elsewhere)

// Optional: remove if you don't need I/O in this header
// #include <iostream>    

namespace dropout {

    // Assumes the Numeric concept is already defined in another header:
    // template <typename T>
    // concept Numeric = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

    // Data structure for managing dropout parameters and state
    // aligned to 64 bytes for cache efficiency
    template <Numeric T>
    struct alignas(64) DropoutData {
        std::vector<T> values;                // Values after dropout is applied
        std::vector<T> mask;                  // Dropout mask (0/1 or similar)
        std::vector<std::mt19937> generators; // Per-thread random number generators
        T dropout_rate{ 0.5 };                 // Probability of dropping a value
        T scale_factor{ 1.0 };                 // Scaling factor to maintain expected sum
        bool is_training{ true };              // Whether dropout is currently active
        mutable std::atomic<size_t> access_counter{ 0 }; // Thread-safe counter

        // Constructor
        explicit DropoutData(size_t size, T rate = T(0.5))
            : values(size), mask(size), dropout_rate(rate)
        {
            const auto num_threads = std::thread::hardware_concurrency();
            generators.reserve(num_threads);

            std::random_device rd;
            for (size_t i = 0; i < num_threads; ++i) {
                generators.emplace_back(rd() + i); // Initialize RNG for each thread
            }

            update_scale_factor();
        }

        // Updates the scale factor based on training state and dropout rate
        void update_scale_factor() noexcept {
            scale_factor = is_training ? (T(1) / (T(1) - dropout_rate)) : T(1);
        }

        // Resizes the value and mask vectors
        void resize(size_t new_size) {
            values.resize(new_size);
            mask.resize(new_size);
        }
    };

} // namespace dropout
