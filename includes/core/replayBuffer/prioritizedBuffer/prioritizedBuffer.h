#pragma once
// Standard and STL headers used for various utilities
#include <iostream>
#include <vector>
#include <random>
#include <deque>
#include <algorithm>
#include <memory>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <concepts>
#include <ranges>
#include <execution>
#include <tuple>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <future>
#include <atomic>
#include <type_traits>
#include <numeric>
#include <cmath>
#include <functional>
#include <span>

namespace fs = std::filesystem;

// Metaprogramming - SFINAE to check if a type is serializable
template<typename T, typename = void>
struct is_serializable : std::false_type {};

template<typename T>
struct is_serializable<T, std::void_t<
    decltype(std::declval<std::ostream&>() << std::declval<T>()),
    decltype(std::declval<std::istream&>() >> std::declval<T&>())
    >> : std::true_type {};

template<typename T>
inline constexpr bool is_serializable_v = is_serializable<T>::value;

// Concept definition for a generic Environment interface
template<typename Env>
concept Environment = requires(Env env, typename Env::State state, typename Env::Action action) {
    { env.reset() } -> std::same_as<typename Env::State>;
    { env.step(action) } -> std::same_as<std::tuple<typename Env::State, float, bool, std::unordered_map<std::string, float>>>;
    { env.validActions(state) } -> std::same_as<std::vector<typename Env::Action>>;
    { env.getStateSize() } -> std::same_as<size_t>;
    { env.getActionSize() } -> std::same_as<size_t>;
    { env.render() } -> std::same_as<void>;
};

// Execution policy traits to identify and name the policy type
template<typename Policy>
struct ExecutionPolicyTraits;

template<>
struct ExecutionPolicyTraits<std::execution::sequenced_policy> {
    static constexpr bool is_parallel = false;
    static constexpr const char* name = "sequential";
};

template<>
struct ExecutionPolicyTraits<std::execution::parallel_policy> {
    static constexpr bool is_parallel = true;
    static constexpr const char* name = "parallel";
};

template<>
struct ExecutionPolicyTraits<std::execution::parallel_unsequenced_policy> {
    static constexpr bool is_parallel = true;
    static constexpr const char* name = "parallel_unsequenced";
};

// A structure representing one experience step, with optional serialization
template <typename State, typename Action>
struct Experience {
    State state;
    Action action;
    float reward;
    State nextState;
    bool done;
    std::unordered_map<std::string, float> info;

    // Conditional serialization using SFINAE
    template<typename S = State, typename A = Action>
    std::enable_if_t<is_serializable_v<S>&& is_serializable_v<A>, void>
        serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&reward), sizeof(reward));
        os.write(reinterpret_cast<const char*>(&done), sizeof(done));
        // Further serialization for state, action, etc.
    }

    template<typename S = State, typename A = Action>
    std::enable_if_t<is_serializable_v<S>&& is_serializable_v<A>, void>
        deserialize(std::istream& is) {
        is.read(reinterpret_cast<char*>(&reward), sizeof(reward));
        is.read(reinterpret_cast<char*>(&done), sizeof(done));
        // Further deserialization
    }
};

// Configuration struct for the buffer, allows flexible customization
template<
    typename State,
    typename Action,
    size_t ChunkSize = 1024,
    bool UseThreading = true,
    bool UsePersistence = false,
    typename ExecPolicy = std::execution::parallel_policy
>
struct BufferConfig {
    using StateType = State;
    using ActionType = Action;
    static constexpr size_t chunkSize = ChunkSize;
    static constexpr bool useThreading = UseThreading;
    static constexpr bool usePersistence = UsePersistence;
    using ExecutionPolicy = ExecPolicy;
};

// Main prioritized experience replay buffer class with multithreading support
template <typename State, typename Action, typename Config = BufferConfig<State, Action>>
class PrioritizedBuffer {
public:
    using experience_type = Experience<State, Action>;
    using priority_type = float;
    using buffer_item = std::pair<experience_type, priority_type>;
    using chunk_type = std::vector<buffer_item>;

    // Constructor with default parameters and dynamic thread count
    PrioritizedBuffer(
        size_t size,
        float alpha = 0.6f,
        float beta = 0.4f,
        float betaIncrement = 0.001f,
        float epsilon = 1e-6f,
        size_t numThreads = std::thread::hardware_concurrency()
    );
    ~PrioritizedBuffer();

    // Standard methods for adding experience
    void add(experience_type& exp, float priority = 0.1f);
    void add(experience_type&& exp, float priority = 0.1f); // Overload for rvalue

    // Batch addition of experiences for performance
    template<typename Iterator>
    void addBatch(Iterator begin, Iterator end, float defaultPriority = 0.1f);

    // Sampling method supporting multithreading
    std::tuple<std::vector<experience_type>, std::vector<size_t>, std::vector<float>>
        sample(size_t batchSize);

    // Asynchronous sampling
    std::future<std::tuple<std::vector<experience_type>, std::vector<size_t>, std::vector<float>>>
        sampleAsync(size_t batchSize);

    // Update priorities with async support
    void updatePriorities(const std::vector<size_t>& indices, const std::vector<float>& newPriorities);
    std::future<void> updatePrioritiesAsync(std::vector<size_t> indices, std::vector<float> newPriorities);

    // Utility methods for capacity and status
    size_t size() const;
    bool empty() const;
    void clear();
    void reserve(size_t capacity);

    // Persistence support (enabled only when UsePersistence = true)
    template<bool UsePersistence = Config::usePersistence>
    std::enable_if_t<UsePersistence, bool> saveToFile(const std::string& filename);

    template<bool UsePersistence = Config::usePersistence>
    std::enable_if_t<UsePersistence, bool> loadFromFile(const std::string& filename);

    // Advanced methods for memory and performance
    void optimizeMemory();
    void prefetchNextBatch(size_t batchSize);

    // Dynamically change execution policy or thread count
    void setExecutionPolicy(bool useParallel);
    void setNumThreads(size_t numThreads);

private:
    // Core data structure for storing experience chunks
    std::vector<chunk_type> bufferChunks;
    std::vector<float> priorities;
    std::atomic<size_t> currentSize{ 0 };
    size_t maxSize;
    float alpha;
    std::atomic<float> beta;
    float betaIncrement;
    float epsilon;

    // Multithreading management
    mutable std::shared_mutex bufferMutex;
    std::vector<std::thread> workerThreads;
    std::atomic<bool> isRunning{ true };
    size_t numThreads;

    // RNG state protected by mutex
    mutable std::mutex rngMutex;
    std::mt19937 rng;

    // Cache for improving performance
    std::atomic<float> sumPriorities{ 0.0f };
    std::vector<size_t> prefetchedIndices;
    std::vector<float> prefetchedWeights;

    // Execution policy state
    bool useParallelExecution;

    // Private helper methods
    float calculateWeight(float probability, size_t totalSize) const;
    std::vector<float> calculateProbabilities();
    size_t findReplacementIndex();
    void rebalanceChunks();
    void prefetchDataInternal(size_t batchSize);

    // Worker thread logic
    void workerFunction();
};




