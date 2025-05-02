
#include <numeric> 
#include <cmath>   
#include "PrioritizedBuffer.h"


// Implementation of the constructor
// Creates a prioritized experience replay buffer with specified parameters
template <typename State, typename Action, typename Config>
PrioritizedBuffer<State, Action, Config>::PrioritizedBuffer(
    size_t size, float alpha, float beta, float betaIncrement, float epsilon, size_t numThreads)
    : maxSize(size), alpha(alpha), beta(beta), betaIncrement(betaIncrement),
    epsilon(epsilon), numThreads(numThreads),
    useParallelExecution(ExecutionPolicyTraits<typename Config::ExecutionPolicy>::is_parallel) {

    // Initial allocation - optimized for memory performance
    // Calculate required chunks based on chunk size configuration
    size_t numChunks = (size + Config::chunkSize - 1) / Config::chunkSize;
    bufferChunks.resize(numChunks);
    for (auto& chunk : bufferChunks) {
        chunk.reserve(Config::chunkSize);
    }

    priorities.reserve(size);

    // Initialize random number generator
    std::random_device rd;
    rng = std::mt19937(rd());

    // Start worker threads if threading is enabled in configuration
    if constexpr (Config::useThreading) {
        for (size_t i = 0; i < numThreads; ++i) {
            workerThreads.emplace_back(&PrioritizedBuffer::workerFunction, this);
        }
    }

    // Preallocate memory for prefetched data to improve performance
    prefetchedIndices.reserve(size / 4);
    prefetchedWeights.reserve(size / 4);
}

// Destructor - ensures proper cleanup of worker threads
template <typename State, typename Action, typename Config>
PrioritizedBuffer<State, Action, Config>::~PrioritizedBuffer() {
    if constexpr (Config::useThreading) {
        isRunning = false;
        for (auto& thread : workerThreads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }
}

// Add experience to buffer with optimized fine-grained locking
// Takes an experience reference and its priority value
template <typename State, typename Action, typename Config>
void PrioritizedBuffer<State, Action, Config>::add(experience_type& exp, float priority) {
    float newPriority = std::pow(priority + epsilon, alpha);

    std::unique_lock<std::shared_mutex> lock(bufferMutex);

    if (currentSize >= maxSize) {
        // Buffer is full, find index to replace using strategy
        size_t idx = findReplacementIndex();
        size_t chunkIdx = idx / Config::chunkSize;
        size_t itemIdx = idx % Config::chunkSize;

        // Update atomic sum before modification
        sumPriorities.fetch_sub(priorities[idx]);
        sumPriorities.fetch_add(newPriority);

        bufferChunks[chunkIdx][itemIdx] = { exp, newPriority };
        priorities[idx] = newPriority;
    }
    else {
        // Add to the last chunk
        size_t chunkIdx = currentSize / Config::chunkSize;
        size_t itemIdx = currentSize % Config::chunkSize;

        if (itemIdx == 0 && chunkIdx >= bufferChunks.size()) {
            bufferChunks.emplace_back();
            bufferChunks.back().reserve(Config::chunkSize);
        }

        bufferChunks[chunkIdx].push_back({ exp, newPriority });
        priorities.push_back(newPriority);

        // Atomic update
        sumPriorities.fetch_add(newPriority);
        currentSize.fetch_add(1);
    }
}

// Overload for rvalue (move semantics)
// More efficient when passing temporary experience objects
template <typename State, typename Action, typename Config>
void PrioritizedBuffer<State, Action, Config>::add(experience_type&& exp, float priority) {
    float newPriority = std::pow(priority + epsilon, alpha);

    std::unique_lock<std::shared_mutex> lock(bufferMutex);

    if (currentSize >= maxSize) {
        size_t idx = findReplacementIndex();
        size_t chunkIdx = idx / Config::chunkSize;
        size_t itemIdx = idx % Config::chunkSize;

        sumPriorities.fetch_sub(priorities[idx]);
        sumPriorities.fetch_add(newPriority);

        bufferChunks[chunkIdx][itemIdx] = { std::move(exp), newPriority };
        priorities[idx] = newPriority;
    }
    else {
        size_t chunkIdx = currentSize / Config::chunkSize;
        size_t itemIdx = currentSize % Config::chunkSize;

        if (itemIdx == 0 && chunkIdx >= bufferChunks.size()) {
            bufferChunks.emplace_back();
            bufferChunks.back().reserve(Config::chunkSize);
        }

        bufferChunks[chunkIdx].push_back({ std::move(exp), newPriority });
        priorities.push_back(newPriority);

        sumPriorities.fetch_add(newPriority);
        currentSize.fetch_add(1);
    }
}

// Batch addition to amortize overhead costs
// Processes multiple experiences at once for better performance
template <typename State, typename Action, typename Config>
template <typename Iterator>
void PrioritizedBuffer<State, Action, Config>::addBatch(Iterator begin, Iterator end, float defaultPriority) {
    // Multi-threaded processing for large quantities of experiences
    if constexpr (Config::useThreading && std::is_same_v<typename Config::ExecutionPolicy, std::execution::parallel_policy>) {
        std::vector<std::future<void>> futures;
        size_t batchSize = std::distance(begin, end);
        size_t perThreadBatch = batchSize / numThreads;

        for (size_t i = 0; i < numThreads; ++i) {
            auto batchBegin = begin + i * perThreadBatch;
            auto batchEnd = (i == numThreads - 1) ? end : batchBegin + perThreadBatch;

            futures.push_back(std::async(std::launch::async, [this, batchBegin, batchEnd, defaultPriority]() {
                for (auto it = batchBegin; it != batchEnd; ++it) {
                    this->add(*it, defaultPriority);
                }
                }));
        }

        // Wait for all threads to complete
        for (auto& future : futures) {
            future.wait();
        }
    }
    else {
        // Sequential version for small quantities
        for (auto it = begin; it != end; ++it) {
            add(*it, defaultPriority);
        }
    }
}

// Optimized sampling with conditional parallelism
// Returns a batch of experiences based on their priorities
template <typename State, typename Action, typename Config>
std::tuple<std::vector<typename PrioritizedBuffer<State, Action, Config>::experience_type>,
    std::vector<size_t>,
    std::vector<float>>
    PrioritizedBuffer<State, Action, Config>::sample(size_t batchSize) {
    std::vector<experience_type> batch;
    std::vector<size_t> indices;
    std::vector<float> weights;

    if (currentSize == 0) {
        return { batch, indices, weights };
    }

    batch.reserve(batchSize);
    indices.reserve(batchSize);
    weights.reserve(batchSize);

    // Shared read lock - multiple threads can read simultaneously
    std::shared_lock<std::shared_mutex> lock(bufferMutex);

    // Use precalculated values if available
    float totalPriorities = sumPriorities.load();

    // Calculate probabilities efficiently
    std::vector<float> probability;
    probability.reserve(currentSize);

    // Local collection to calculate max in parallel
    float maxWeight = 0.0f;

    // Create distribution for efficient sampling
    {
        std::lock_guard<std::mutex> rngLock(rngMutex);

        // Parallelize probability calculation if appropriate
        if constexpr (ExecutionPolicyTraits<typename Config::ExecutionPolicy>::is_parallel) {
            for (size_t i = 0; i < currentSize; ++i) {
                size_t chunkIdx = i / Config::chunkSize;
                size_t itemIdx = i % Config::chunkSize;
                probability.push_back(bufferChunks[chunkIdx][itemIdx].second / totalPriorities);
            }
        }
        else {
            for (const auto& p : priorities) {
                probability.push_back(p / totalPriorities);
            }
        }

        std::discrete_distribution<size_t> dist(probability.begin(), probability.end());

        for (size_t i = 0; i < batchSize; ++i) {
            size_t idx = dist(rng);
            indices.push_back(idx);

            size_t chunkIdx = idx / Config::chunkSize;
            size_t itemIdx = idx % Config::chunkSize;

            batch.push_back(bufferChunks[chunkIdx][itemIdx].first);

            float weight = calculateWeight(probability[idx], currentSize);
            weights.push_back(weight);
            maxWeight = std::max(maxWeight, weight);
        }
    }

    // Normalize weights for stability in training
    if (maxWeight > 0) {
        if constexpr (ExecutionPolicyTraits<typename Config::ExecutionPolicy>::is_parallel) {
            std::transform(
                std::execution::par_unseq,
                weights.begin(), weights.end(),
                weights.begin(),
                [maxWeight](float w) { return w / maxWeight; }
            );
        }
        else {
            for (auto& w : weights) {
                w /= maxWeight;
            }
        }
    }

    // Update beta in a thread-safe manner
    beta.store(std::min(1.0f, beta.load() + betaIncrement));

    return { batch, indices, weights };
}

// Asynchronous version of sample
// Returns a future to allow non-blocking operation
template <typename State, typename Action, typename Config>
std::future<std::tuple<std::vector<typename PrioritizedBuffer<State, Action, Config>::experience_type>,
    std::vector<size_t>,
    std::vector<float>>>
    PrioritizedBuffer<State, Action, Config>::sampleAsync(size_t batchSize) {
    return std::async(std::launch::async, [this, batchSize]() {
        return this->sample(batchSize);
        });
}

// Parallelized priority updates
// Updates priorities for multiple experiences at once
template <typename State, typename Action, typename Config>
void PrioritizedBuffer<State, Action, Config>::updatePriorities(
    const std::vector<size_t>& indices,
    const std::vector<float>& newPriorities
) {
    std::unique_lock<std::shared_mutex> lock(bufferMutex);

    if constexpr (ExecutionPolicyTraits<typename Config::ExecutionPolicy>::is_parallel) {
        std::for_each(
            std::execution::par_unseq,
            indices.begin(), indices.end(),
            [this, &indices, &newPriorities](size_t i) {
                size_t idx = indices[i];
                float oldPriority = priorities[idx];
                float newPriority = std::pow(newPriorities[i] + epsilon, alpha);

                size_t chunkIdx = idx / Config::chunkSize;
                size_t itemIdx = idx % Config::chunkSize;

                bufferChunks[chunkIdx][itemIdx].second = newPriority;
                priorities[idx] = newPriority;

                // Atomic update of total
                sumPriorities.fetch_add(newPriority - oldPriority);
            }
        );
    }
    else {
        for (size_t i = 0; i < indices.size(); ++i) {
            size_t idx = indices[i];
            if (idx >= currentSize) continue;

            float oldPriority = priorities[idx];
            float newPriority = std::pow(newPriorities[i] + epsilon, alpha);

            size_t chunkIdx = idx / Config::chunkSize;
            size_t itemIdx = idx % Config::chunkSize;

            bufferChunks[chunkIdx][itemIdx].second = newPriority;
            priorities[idx] = newPriority;

            sumPriorities.fetch_add(newPriority - oldPriority);
        }
    }
}

// Asynchronous priority updates
// Non-blocking version of updatePriorities
template <typename State, typename Action, typename Config>
std::future<void> PrioritizedBuffer<State, Action, Config>::updatePrioritiesAsync(
    std::vector<size_t> indices,
    std::vector<float> newPriorities
) {
    return std::async(std::launch::async, [this, indices = std::move(indices), newPriorities = std::move(newPriorities)]() {
        this->updatePriorities(indices, newPriorities);
        });
}

// Get current buffer size
template <typename State, typename Action, typename Config>
size_t PrioritizedBuffer<State, Action, Config>::size() const {
    return currentSize.load();
}

// Check if buffer is empty
template <typename State, typename Action, typename Config>
bool PrioritizedBuffer<State, Action, Config>::empty() const {
    return currentSize.load() == 0;
}

// Clear all buffer data
template <typename State, typename Action, typename Config>
void PrioritizedBuffer<State, Action, Config>::clear() {
    std::unique_lock<std::shared_mutex> lock(bufferMutex);

    for (auto& chunk : bufferChunks) {
        chunk.clear();
    }
    priorities.clear();
    currentSize.store(0);
    sumPriorities.store(0.0f);
}

// Reserve space for buffer expansion
template <typename State, typename Action, typename Config>
void PrioritizedBuffer<State, Action, Config>::reserve(size_t capacity) {
    std::unique_lock<std::shared_mutex> lock(bufferMutex);

    size_t numChunks = (capacity + Config::chunkSize - 1) / Config::chunkSize;
    bufferChunks.resize(numChunks);
    for (auto& chunk : bufferChunks) {
        chunk.reserve(Config::chunkSize);
    }
    priorities.reserve(capacity);
}

// Save buffer to file (conditional implementation)
// Only available if persistence is enabled in configuration
template <typename State, typename Action, typename Config>
template <bool UsePersistence>
std::enable_if_t<UsePersistence, bool>
PrioritizedBuffer<State, Action, Config>::saveToFile(const std::string& filename) {
    std::shared_lock<std::shared_mutex> lock(bufferMutex);

    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;

    // Write header information
    size_t size = currentSize.load();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    file.write(reinterpret_cast<const char*>(&alpha), sizeof(alpha));
    file.write(reinterpret_cast<const char*>(&beta), sizeof(beta));
    file.write(reinterpret_cast<const char*>(&betaIncrement), sizeof(betaIncrement));
    file.write(reinterpret_cast<const char*>(&epsilon), sizeof(epsilon));

    // Write data in parallel if possible
    for (size_t i = 0; i < size; ++i) {
        size_t chunkIdx = i / Config::chunkSize;
        size_t itemIdx = i % Config::chunkSize;

        if constexpr (is_serializable_v<State> && is_serializable_v<Action>) {
            bufferChunks[chunkIdx][itemIdx].first.serialize(file);
        }

        file.write(reinterpret_cast<const char*>(&priorities[i]), sizeof(float));
    }

    return file.good();
}

// Load buffer from file (conditional implementation)
// Only available if persistence is enabled in configuration
template <typename State, typename Action, typename Config>
template <bool UsePersistence>
std::enable_if_t<UsePersistence, bool>
PrioritizedBuffer<State, Action, Config>::loadFromFile(const std::string& filename) {
    std::unique_lock<std::shared_mutex> lock(bufferMutex);

    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;

    // Read header information
    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    file.read(reinterpret_cast<char*>(&alpha), sizeof(alpha));
    float betaValue;
    file.read(reinterpret_cast<char*>(&betaValue), sizeof(betaValue));
    beta.store(betaValue);
    file.read(reinterpret_cast<char*>(&betaIncrement), sizeof(betaIncrement));
    file.read(reinterpret_cast<char*>(&epsilon), sizeof(epsilon));

    // Clear existing data
    clear();

    // Preallocate space
    reserve(size);

    // Read data
    float sumPrioritiesValue = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        experience_type exp;
        if constexpr (is_serializable_v<State> && is_serializable_v<Action>) {
            exp.deserialize(file);
        }

        float priority;
        file.read(reinterpret_cast<char*>(&priority), sizeof(float));

        size_t chunkIdx = i / Config::chunkSize;
        size_t itemIdx = i % Config::chunkSize;

        if (chunkIdx >= bufferChunks.size()) {
            bufferChunks.emplace_back();
            bufferChunks.back().reserve(Config::chunkSize);
        }

        bufferChunks[chunkIdx].push_back({ exp, priority });
        priorities.push_back(priority);
        sumPrioritiesValue += priority;
    }

    currentSize.store(size);
    sumPriorities.store(sumPrioritiesValue);

    return file.good();
}

// Memory optimization for the buffer
// Compacts and reorganizes chunks to minimize memory usage
template <typename State, typename Action, typename Config>
void PrioritizedBuffer<State, Action, Config>::optimizeMemory() {
    std::unique_lock<std::shared_mutex> lock(bufferMutex);

    // Recalculate necessary number of chunks
    size_t numChunks = (currentSize + Config::chunkSize - 1) / Config::chunkSize;

    // Compact data if necessary
    if (numChunks < bufferChunks.size()) {
        // Move data to compact chunks
        for (size_t i = 0; i < currentSize; ++i) {
            size_t oldChunkIdx = i / Config::chunkSize;
            size_t oldItemIdx = i % Config::chunkSize;

            size_t newChunkIdx = i / Config::chunkSize;
            size_t newItemIdx = i % Config::chunkSize;

            if (oldChunkIdx != newChunkIdx || oldItemIdx != newItemIdx) {
                bufferChunks[newChunkIdx][newItemIdx] = std::move(bufferChunks[oldChunkIdx][oldItemIdx]);
            }
        }

        // Reduce number of chunks
        bufferChunks.resize(numChunks);
    }

    // Free excess memory
    for (auto& chunk : bufferChunks) {
        chunk.shrink_to_fit();
    }
    bufferChunks.shrink_to_fit();
    priorities.shrink_to_fit();
}

// Prefetch data for the next batch
// Prepares data in advance to reduce latency
template <typename State, typename Action, typename Config>
void PrioritizedBuffer<State, Action, Config>::prefetchNextBatch(size_t batchSize) {
    if constexpr (Config::useThreading) {
        std::async(std::launch::async, [this, batchSize]() {
            this->prefetchDataInternal(batchSize);
            });
    }
    else {
        prefetchDataInternal(batchSize);
    }
}

// Configure execution policy
// Enables switching between parallel and sequential execution
template <typename State, typename Action, typename Config>
void PrioritizedBuffer<State, Action, Config>::setExecutionPolicy(bool useParallel) {
    useParallelExecution = useParallel;
}

// Configure number of worker threads
// Adjusts threading based on available hardware
template <typename State, typename Action, typename Config>
void PrioritizedBuffer<State, Action, Config>::setNumThreads(size_t threads) {
    if constexpr (Config::useThreading) {
        // Stop existing worker threads
        isRunning = false;
        for (auto& thread : workerThreads) {
            if (thread.joinable()) {
                thread.join();
            }
        }

        // Restart with new thread count
        numThreads = threads;
        workerThreads.clear();
        isRunning = true;

        for (size_t i = 0; i < numThreads; ++i) {
            workerThreads.emplace_back(&PrioritizedBuffer::workerFunction, this);
        }
    }
}

// Private method to calculate importance sampling weight
// Adjusts weights based on beta parameter
template <typename State, typename Action, typename Config>
float PrioritizedBuffer<State, Action, Config>::calculateWeight(float probability, size_t totalSize) const {
    return std::pow(totalSize * probability, -beta.load());
}

// Private method to calculate probabilities
// Converts priorities to sampling probabilities
template <typename State, typename Action, typename Config>
std::vector<float> PrioritizedBuffer<State, Action, Config>::calculateProbabilities() {
    std::vector<float> probability;
    probability.reserve(currentSize);

    float totalPriorities = sumPriorities.load();
    if (totalPriorities == 0.0f) {
        totalPriorities = 1e-8f;  // Avoid division by zero
    }

    if constexpr (ExecutionPolicyTraits<typename Config::ExecutionPolicy>::is_parallel) {
        size_t size = currentSize.load();
        probability.resize(size);

        std::transform(
            std::execution::par_unseq,
            priorities.begin(), priorities.begin() + size,
            probability.begin(),
            [totalPriorities](float p) { return p / totalPriorities; }
        );
    }
    else {
        for (size_t i = 0; i < currentSize; ++i) {
            probability.push_back(priorities[i] / totalPriorities);
        }
    }

    return probability;
}

// Find index to replace (min-priority strategy)
// Identifies lowest priority experience for replacement
template <typename State, typename Action, typename Config>
size_t PrioritizedBuffer<State, Action, Config>::findReplacementIndex() {
    // Use parallelization to find minimum faster
    if constexpr (ExecutionPolicyTraits<typename Config::ExecutionPolicy>::is_parallel) {
        auto minIt = std::min_element(
            std::execution::par_unseq,
            priorities.begin(), priorities.begin() + currentSize
        );
        return std::distance(priorities.begin(), minIt);
    }
    else {
        auto minIt = std::min_element(priorities.begin(), priorities.begin() + currentSize);
        return std::distance(priorities.begin(), minIt);
    }
}

// Rebalance chunks to optimize memory access
// Groups frequently accessed elements together
template <typename State, typename Action, typename Config>
void PrioritizedBuffer<State, Action, Config>::rebalanceChunks() {
    // This method can be called periodically to optimize memory access
    // by grouping frequently accessed elements in the same chunks

    // If buffer is almost empty, no action needed
    if (currentSize < Config::chunkSize) {
        return;
    }

    // Sort experiences by priority to group similar priorities
    std::vector<std::pair<size_t, float>> indexedPriorities;
    indexedPriorities.reserve(currentSize);

    for (size_t i = 0; i < currentSize; ++i) {
        indexedPriorities.push_back({ i, priorities[i] });
    }

    // Sort by descending priority
    if constexpr (ExecutionPolicyTraits<typename Config::ExecutionPolicy>::is_parallel) {
        std::sort(
            std::execution::par_unseq,
            indexedPriorities.begin(), indexedPriorities.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; }
        );
    }
    else {
        std::sort(
            indexedPriorities.begin(), indexedPriorities.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; }
        );
    }

    // Create new buffer structure
    std::vector<chunk_type> newChunks(bufferChunks.size());
    std::vector<float> newPriorities(currentSize);

    // Reorganize data
    for (size_t i = 0; i < currentSize; ++i) {
        size_t oldIdx = indexedPriorities[i].first;
        size_t oldChunkIdx = oldIdx / Config::chunkSize;
        size_t oldItemIdx = oldIdx % Config::chunkSize;

        size_t newChunkIdx = i / Config::chunkSize;
        size_t newItemIdx = i % Config::chunkSize;

        if (newChunkIdx >= newChunks.size()) {
            newChunks.emplace_back();
        }

        if (newItemIdx == 0) {
            newChunks[newChunkIdx].reserve(Config::chunkSize);
        }

        newChunks[newChunkIdx].push_back(bufferChunks[oldChunkIdx][oldItemIdx]);
        newPriorities[i] = priorities[oldIdx];
    }

    // Replace old structures
    bufferChunks = std::move(newChunks);
    priorities = std::move(newPriorities);
}

// Internal data prefetching
// Prepares data in CPU cache for future access
template <typename State, typename Action, typename Config>
void PrioritizedBuffer<State, Action, Config>::prefetchDataInternal(size_t batchSize) {
    std::shared_lock<std::shared_mutex> lock(bufferMutex);

    if (currentSize == 0) {
        return;
    }

    std::vector<size_t> indices;
    std::vector<float> weights;
    indices.reserve(batchSize);
    weights.reserve(batchSize);

    float totalPriorities = sumPriorities.load();
    auto probability = calculateProbabilities();

    std::lock_guard<std::mutex> rngLock(rngMutex);
    std::discrete_distribution<size_t> dist(probability.begin(), probability.end());

    for (size_t i = 0; i < batchSize; ++i) {
        size_t idx = dist(rng);
        indices.push_back(idx);

        // Prefetch data into CPU cache
        size_t chunkIdx = idx / Config::chunkSize;
        size_t itemIdx = idx % Config::chunkSize;

        // Prefetching technique that signals CPU to load data
        // into cache for fast future access
        __builtin_prefetch(&bufferChunks[chunkIdx][itemIdx], 0, 3);

        float weight = calculateWeight(probability[idx], currentSize);
        weights.push_back(weight);
    }

    // Store precalculated indices and weights
    {
        std::unique_lock<std::shared_mutex> writeLock(bufferMutex);
        prefetchedIndices = std::move(indices);
        prefetchedWeights = std::move(weights);
    }
}

// Worker thread function
// Performs background tasks for optimization
template <typename State, typename Action, typename Config>
void PrioritizedBuffer<State, Action, Config>::workerFunction() {
    while (isRunning) {
        // Perform background tasks like prefetching,
        // reorganization or other optimizations

        // The following task is executed periodically
        if (currentSize > Config::chunkSize) {
            // Recalculate total priority sum to avoid drift
            float calculatedSum = 0.0f;
            {
                std::shared_lock<std::shared_mutex> lock(bufferMutex);
                calculatedSum = std::accumulate(
                    priorities.begin(),
                    priorities.begin() + currentSize,
                    0.0f
                );
            }

            // Update atomic sum if needed
            float oldSum = sumPriorities.load();
            if (std::abs(oldSum - calculatedSum) > epsilon) {
                sumPriorities.store(calculatedSum);
            }
        }

        // Pause to reduce CPU usage
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// Explicit instantiation for common types
// Helps with compilation time and ensures template code is generated
template class PrioritizedBuffer<std::vector<float>, int>;
template class PrioritizedBuffer<std::vector<float>, int, BufferConfig<std::vector<float>, int, 2048, true>>;
template class PrioritizedBuffer<std::vector<float>, int, BufferConfig<std::vector<float>, int, 4096, true, true>>;

// Equality operator for Experience
// Used for comparing experiences in testing
template <typename State, typename Action>
bool operator==(const Experience<State, Action>& lhs, const Experience<State, Action>& rhs) {
    return lhs.state == rhs.state &&
        lhs.action == rhs.action &&
        std::abs(lhs.reward - rhs.reward) < 1e-6f &&
        lhs.nextState == rhs.nextState &&
        lhs.done == rhs.done;
}

// Utility functions

// Factory for creating buffers with different configurations
// Simplifies creation of optimized buffers based on use case
template <typename State, typename Action>
auto createOptimizedBuffer(size_t size, bool useThreading, bool usePermanentStorage) {
    if (useThreading) {
        if (usePermanentStorage) {
            return PrioritizedBuffer<State, Action, BufferConfig<State, Action, 4096, true, true>>(size);
        }
        else {
            return PrioritizedBuffer<State, Action, BufferConfig<State, Action, 4096, true, false>>(size);
        }
    }
    else {
        return PrioritizedBuffer<State, Action, BufferConfig<State, Action, 4096, false, false>>(size);
    }
}

