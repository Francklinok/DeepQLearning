#pragma once
#include <mutex>
#include <random>
#include <unordered_map>
#include <string>
#include <shared_mutex>
#include <thread>

/**
 * @brief Execution context providing random number generation, parallel settings, and caching mechanism
 * @tparam T Data type (e.g., float or double)
 */
template <typename T = float>
class ExecutionContext {
public:
    /**
     * @brief Constructor
     * @param enableParallel Enables or disables parallel execution
     * @param threads Number of threads to use (-1 for hardware concurrency)
     */
    ExecutionContext(bool enableParallel = true, int threads = -1);

    /**
     * @brief Generates a normally distributed random number
     * @return Random number from normal distribution
     */
    T generateNormalRandom();

    /**
     * @brief Generates a uniformly distributed random number in the given range
     * @param min Minimum value (default = 0)
     * @param max Maximum value (default = 1)
     * @return Random number from uniform distribution
     */
    T generateUniformRandom(T min = 0, T max = 1);

    /**
     * @brief Checks if parallel execution is enabled
     * @return True if parallelism is enabled, false otherwise
     */
    bool isParallelEnabled() const;

    /**
     * @brief Gets the number of threads used for parallel execution
     * @return Number of threads
     */
    int getNumThreads() const;

    /**
     * @brief Retrieves a cached value for a specific step and key
     * @param step Time step
     * @param key Identifier for the cached value
     * @return Cached value if exists
     */
    T getCachedValue(int64_t step, const std::string& key) const;

    /**
     * @brief Sets a cached value for a specific step and key
     * @param step Time step
     * @param key Identifier for the cached value
     * @param value Value to cache
     */
    void setCachedValue(int64_t step, const std::string& key, T value) const;

    /**
     * @brief Clears the entire cache
     */
    void clearCache() const;

private:
    bool parallelExecution;  // Flag for enabling parallel execution
    int numThreads;          // Number of threads used
    mutable std::mutex randomMutex;  // Mutex for random number generation
    std::mt19937 randomEngine;       // Mersenne Twister random engine
    std::normal_distribution<double> normalDist;  // Normal distribution generator
    mutable std::shared_mutex cacheMutex;         // Mutex for thread-safe cache access

    /**
     * @brief Hash function for std::pair (used in unordered_map)
     */
    template <typename A, typename B>
    struct PairHash {
        std::size_t operator()(const std::pair<A, B>& p) const {
            auto h1 = std::hash<A>{}(p.first);
            auto h2 = std::hash<B>{}(p.second);
            return h1 ^ (h2 << 1);  // Combine hashes
        }
    };

    // Thread-safe cache mapping (step, key) pairs to values
    mutable std::unordered_map<std::pair<int64_t, std::string>, T, PairHash<int64_t, std::string>> computeCache;
};
