#pragma once
#include <mutex>
#include <random>
#include <unordered_map>
#include <string>
#include <shared_mutex>
#include <thread>
#include <stdexcept>
#include <utility>

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
    ExecutionContext(bool enableParallel = true, int threads = -1)
        : parallelExecution(enableParallel),
        numThreads(threads > 0 ? threads : static_cast<int>(std::thread::hardware_concurrency())),
        randomEngine(std::random_device{}()),
        normalDist(0.0, 1.0) {
    }

    /**
     * @brief Generates a normally distributed random number
     * @return Random number from normal distribution
     */
    T generateNormalRandom() {
        std::lock_guard<std::mutex> lock(randomMutex);
        return static_cast<T>(normalDist(randomEngine));
    }

    /**
     * @brief Generates a uniformly distributed random number in the given range
     * @param min Minimum value (default = 0)
     * @param max Maximum value (default = 1)
     * @return Random number from uniform distribution
     */
    T generateUniformRandom(T min = static_cast<T>(0), T max = static_cast<T>(1)) {
        std::lock_guard<std::mutex> lock(randomMutex);
        std::uniform_real_distribution<double> dist(static_cast<double>(min), static_cast<double>(max));
        return static_cast<T>(dist(randomEngine));
    }

    /**
     * @brief Checks if parallel execution is enabled
     * @return True if parallelism is enabled, false otherwise
     */
    bool isParallelEnabled() const {
        return parallelExecution;
    }

    /**
     * @brief Gets the number of threads used for parallel execution
     * @return Number of threads
     */
    int getNumThreads() const {
        return numThreads;
    }

    /**
     * @brief Retrieves a cached value for a specific step and key
     * @param step Time step
     * @param key Identifier for the cached value
     * @return Cached value if exists, NaN if not found
     */
    T getCachedValue(int64_t step, const std::string& key) const {
        std::shared_lock<std::shared_mutex> lock(cacheMutex);
        auto it = computeCache.find(std::make_pair(step, key));
        if (it != computeCache.end()) {
            return it->second;
        }
        // Return NaN instead of throwing exception for better performance
        return std::numeric_limits<T>::quiet_NaN();
    }

    /**
     * @brief Sets a cached value for a specific step and key
     * @param step Time step
     * @param key Identifier for the cached value
     * @param value Value to cache
     */
    void setCachedValue(int64_t step, const std::string& key, T value) const {
        std::unique_lock<std::shared_mutex> lock(cacheMutex);
        computeCache[std::make_pair(step, key)] = value;
    }

    /**
     * @brief Clears the entire cache
     */
    void clearCache() const {
        std::unique_lock<std::shared_mutex> lock(cacheMutex);
        computeCache.clear();
    }

    /**
     * @brief Gets the current cache size
     * @return Number of cached entries
     */
    std::size_t getCacheSize() const {
        std::shared_lock<std::shared_mutex> lock(cacheMutex);
        return computeCache.size();
    }

    /**
     * @brief Checks if a value exists in cache
     * @param step Time step
     * @param key Identifier for the cached value
     * @return True if value exists in cache
     */
    bool hasCachedValue(int64_t step, const std::string& key) const {
        std::shared_lock<std::shared_mutex> lock(cacheMutex);
        return computeCache.find(std::make_pair(step, key)) != computeCache.end();
    }

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