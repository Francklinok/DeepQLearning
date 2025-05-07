#include <ExecutionContext.h>


template <typename T>
ExecutionContext<T>::ExecutionContext(bool enableParallel, int threads)
    : parallelExecution(enableParallel),
    numThreads(threads > 0 ? threads : std::thread::hardware_concurrency()),
    randomEngine(std::random_device{}()),
    normalDist(0.0, 1.0) {
}

template <typename T>
T ExecutionContext<T>::generateNormalRandom() {
    std::lock_guard<std::mutex> lock(randomMutex);
    return static_cast<T>(normalDist(randomEngine));
}

template <typename T>
T ExecutionContext<T>::generateUniformRandom(T min, T max) {
    std::lock_guard<std::mutex> lock(randomMutex);
    std::uniform_real_distribution<double> dist(min, max);
    return static_cast<T>(dist(randomEngine));
}

template <typename T>
bool ExecutionContext<T>::isParallelEnabled() const {
    return parallelExecution;
}

template <typename T>
int ExecutionContext<T>::getNumThreads() const {
    return numThreads;
}

template <typename T>
T ExecutionContext<T>::getCachedValue(int64_t step, const std::string& key) const {
    std::shared_lock<std::shared_mutex> lock(cacheMutex);
    auto it = computeCache.find(std::make_pair(step, key));
    if (it != computeCache.end()) {
        return it->second;
    }
    throw std::runtime_error("Value not found in cache");
}

template <typename T>
void ExecutionContext<T>::setCachedValue(int64_t step, const std::string& key, T value) const {
    std::unique_lock<std::shared_mutex> lock(cacheMutex);
    computeCache[std::make_pair(step, key)] = value;
}

template <typename T>
void ExecutionContext<T>::clearCache() const {
    std::unique_lock<std::shared_mutex> lock(cacheMutex);
    computeCache.clear();
}