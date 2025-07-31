#pragma once
#include <vector>
#include <thread>
#include <memory>
#include "concepts.hpp"

namespace deep_qn {
    namespace core {

        // ==================== Layer Data Structure ====================
        struct LayerData {
            std::vector<int> dimensions;
            std::vector<float> weights;
            std::vector<float> biases;
            std::vector<float> noise_params;
            bool is_noisy = false;
            std::string layer_type = "dense";

            // Constructor
            LayerData(int input_size, int output_size, bool noisy = false);

            // Copy constructor
            LayerData(const LayerData& other);

            // Move constructor
            LayerData(LayerData&& other) noexcept;

            // Assignment operators
            LayerData& operator=(const LayerData& other);
            LayerData& operator=(LayerData&& other) noexcept;

            // Destructor
            ~LayerData() = default;

            // Utility methods
            void initialize_weights();
            void initialize_noise_params();
            void clear();
            size_t memory_usage() const;
            bool is_valid() const;

            // Serialization support
            void save_to_buffer(std::vector<uint8_t>& buffer) const;
            void load_from_buffer(const std::vector<uint8_t>& buffer);
        };

        // ==================== Network Configuration ====================
        struct NetworkConfig {
            std::vector<int> hidden_layers;
            float dropout_rate = 0.0f;
            float learning_rate = 0.001f;
            bool batch_norm = true;
            bool layer_norm = false;
            bool use_multithreading = true;
            bool use_noisy_layers = false;
            size_t thread_pool_size = std::thread::hardware_concurrency();
            size_t batch_size = 32;
            size_t max_memory_usage = 1024 * 1024 * 1024; // 1GB default

            // Activation function type
            enum class ActivationType {
                ReLU,
                LeakyReLU,
                ELU,
                Tanh,
                Sigmoid,
                Swish
            };

            ActivationType activation_type = ActivationType::ReLU;

            // Normalization type
            enum class NormalizationType {
                None,
                Batch,
                Layer,
                Group
            };

            NormalizationType normalization_type = NormalizationType::Batch;

            // Initialization method
            enum class InitializationMethod {
                Xavier,
                He,
                LeCun,
                Uniform,
                Normal
            };

            InitializationMethod init_method = InitializationMethod::Xavier;

            // Constructor
            NetworkConfig() = default;

            // Copy constructor
            NetworkConfig(const NetworkConfig& other) = default;

            // Move constructor
            NetworkConfig(NetworkConfig&& other) noexcept = default;

            // Assignment operators
            NetworkConfig& operator=(const NetworkConfig& other) = default;
            NetworkConfig& operator=(NetworkConfig&& other) noexcept = default;

            // Validation
            bool is_valid() const;
            void validate() const;

            // Configuration presets
            static NetworkConfig create_small_network();
            static NetworkConfig create_medium_network();
            static NetworkConfig create_large_network();
            static NetworkConfig create_performance_optimized();
            static NetworkConfig create_memory_optimized();

            // Utility methods
            size_t total_parameters() const;
            size_t estimated_memory_usage() const;
            void optimize_for_hardware();
            void print_summary() const;

            // Serialization
            void save_to_file(const std::string& filename) const;
            void load_from_file(const std::string& filename);
        };

        // ==================== Performance Metrics ====================
        struct PerformanceMetrics {
            size_t forward_pass_count = 0;
            size_t backward_pass_count = 0;
            double total_forward_time = 0.0;
            double total_backward_time = 0.0;
            double average_forward_time = 0.0;
            double average_backward_time = 0.0;
            size_t memory_allocations = 0;
            size_t peak_memory_usage = 0;
            size_t current_memory_usage = 0;

            // Methods
            void reset();
            void update_forward_time(double time);
            void update_backward_time(double time);
            void update_memory_usage(size_t usage);
            void print_summary() const;
            double get_throughput() const;
        };

        // ==================== Training State ====================
        struct TrainingState {
            size_t epoch = 0;
            size_t step = 0;
            double current_loss = 0.0;
            double best_loss = std::numeric_limits<double>::max();
            double current_accuracy = 0.0;
            double best_accuracy = 0.0;
            bool is_training = true;
            bool early_stopping_triggered = false;
            size_t patience_counter = 0;

            // Learning rate schedule
            double initial_learning_rate = 0.001;
            double current_learning_rate = 0.001;
            double learning_rate_decay = 0.95;
            size_t decay_steps = 1000;

            // Methods
            void reset();
            void update_loss(double loss);
            void update_accuracy(double accuracy);
            void update_learning_rate();
            bool should_stop_early(size_t patience = 10) const;
            void save_checkpoint(const std::string& filename) const;
            void load_checkpoint(const std::string& filename);
        };

        // ==================== Memory Pool ====================
        template<typename T>
        class MemoryPool {
        private:
            std::vector<std::unique_ptr<T[]>> pools;
            std::vector<T*> available_blocks;
            size_t block_size;
            size_t pool_size;
            mutable std::mutex mutex_;

        public:
            explicit MemoryPool(size_t block_size, size_t initial_pool_size = 1000);
            ~MemoryPool() = default;

            // Non-copyable
            MemoryPool(const MemoryPool&) = delete;
            MemoryPool& operator=(const MemoryPool&) = delete;

            // Movable
            MemoryPool(MemoryPool&&) noexcept = default;
            MemoryPool& operator=(MemoryPool&&) noexcept = default;

            T* allocate();
            void deallocate(T* ptr);
            void expand_pool();
            size_t available_blocks_count() const;
            size_t total_allocated() const;
            void clear();
        };

    } // namespace core
} // namespace deep_qn