#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <memory>
#include <stdexcept>
#include <numeric>
#include <functional>
#include <algorithm>
#include <cmath>
#include <random>
#include <future>
#include <thread>
#include <utility>
#include <type_traits>
#include <execution>   
#include <immintrin.h> // SIMD intrinsics

#include "tensort_concept.hpp"
#include "general_concepts.hpp"

// namespace pour Tensor
namespace Layer::Tensor {

    template<FloatingPoint T, size_t MaxDims = 4>
    class Tensor {
    public:
        using value_type = T;
        using size_type = size_t;
        using shape_type = std::array<size_type, MaxDims>;

    private:
        alignas(64) std::unique_ptr<T[]> data_;
        shape_type shape_{};
        size_type total_size_{ 0 };
        size_type dims_{ 0 };

        // SIMD width (AVX2)
        static constexpr size_t simd_width = sizeof(__m256) / sizeof(T);

    public:
        Tensor() = default;

        explicit Tensor(std::initializer_list<size_type> shape) {
            reshape(std::vector<size_type>(shape));
        }

        explicit Tensor(const std::vector<size_type>& shape) {
            reshape(shape);
        }

        // Copy constructor
        Tensor(const Tensor& other) {
            *this = other;
        }

        // Move constructor
        Tensor(Tensor&& other) noexcept {
            *this = std::move(other);
        }

        // Copy assignment
        Tensor& operator=(const Tensor& other) {
            if (this != &other) {
                reshape(other.get_shape_vector());
                std::copy_n(other.data(), total_size_, data_.get());
            }
            return *this;
        }

        // Move assignment
        Tensor& operator=(Tensor&& other) noexcept {
            if (this != &other) {
                data_ = std::move(other.data_);
                shape_ = other.shape_;
                total_size_ = other.total_size_;
                dims_ = other.dims_;
                other.reset();
            }
            return *this;
        }

        // Reshape
        void reshape(const std::vector<size_type>& new_shape) {
            if (new_shape.size() > MaxDims) {
                throw std::invalid_argument("Too many dimensions");
            }
            dims_ = new_shape.size();
            shape_.fill(1);
            std::copy(new_shape.begin(), new_shape.end(), shape_.begin());

            total_size_ = std::accumulate(
                new_shape.begin(), new_shape.end(), size_type{ 1 }, std::multiplies<size_type>{});

            size_type aligned_size = (total_size_ + simd_width - 1) & ~(simd_width - 1);
            data_ = std::make_unique<T[]>(aligned_size);
            std::fill_n(data_.get(), aligned_size, T{ 0 });
        }

        // Data access
        T* data() noexcept { return data_.get(); }
        const T* data() const noexcept { return data_.get(); }

        size_type size() const noexcept { return total_size_; }
        size_type dims() const noexcept { return dims_; }

        const shape_type& shape() const noexcept { return shape_; }

        std::vector<size_type> get_shape_vector() const {
            return std::vector<size_type>(shape_.begin(), shape_.begin() + dims_);
        }

        // Element access
        template<typename... Indices>
        T& operator()(Indices... indices) {
            static_assert(sizeof...(indices) <= MaxDims);
#ifdef _DEBUG
            if (sizeof...(indices) != dims_) {
                throw std::invalid_argument("Dimension mismatch");
            }
#endif
            return data_[compute_index(static_cast<size_type>(indices)...)];
        }

        template<typename... Indices>
        const T& operator()(Indices... indices) const {
            static_assert(sizeof...(indices) <= MaxDims);
#ifdef _DEBUG
            if (sizeof...(indices) != dims_) {
                throw std::invalid_argument("Dimension mismatch");
            }
#endif
            return data_[compute_index(static_cast<size_type>(indices)...)];
        }

        // Operators
        Tensor& operator+=(const Tensor& other) {
            simd_add(other);
            return *this;
        }

        Tensor& operator*=(T scalar) {
            simd_multiply(scalar);
            return *this;
        }

        Tensor& apply(const std::function<T(T)>& func) {
            simd_apply(func);
            return *this;
        }

        // Matrix multiplication (2D tensors only)
        Tensor matmul(const Tensor& other) const {
            if (dims_ != 2 || other.dims_ != 2) {
                throw std::invalid_argument("matmul requires 2D tensors");
            }
            if (shape_[1] != other.shape_[0]) {
                throw std::invalid_argument("Incompatible dimensions for matmul");
            }

            Tensor result({ shape_[0], other.shape_[1] });
            simd_matmul(other, result);
            return result;
        }

        // Statistical operations
        T mean() const {
            return std::accumulate(data_.get(), data_.get() + total_size_, T{ 0 })
                / static_cast<T>(total_size_);
        }

        T variance() const {
            T m = mean();
            T sum_sq = std::transform_reduce(
                std::execution::par_unseq,
                data_.get(), data_.get() + total_size_,
                T{ 0 }, std::plus<T>{},
                [m](T x) { return (x - m) * (x - m); });
            return sum_sq / static_cast<T>(total_size_);
        }

        T norm() const {
            T sum_sq = std::transform_reduce(
                std::execution::par_unseq,
                data_.get(), data_.get() + total_size_,
                T{ 0 }, std::plus<T>{},
                [](T x) { return x * x; });
            return std::sqrt(sum_sq);
        }

        // Initialization helpers
        void fill(T value) {
            std::fill_n(data_.get(), total_size_, value);
        }

        void random_normal(T mean = T{ 0 }, T stddev = T{ 1 }) {
            static thread_local std::random_device rd;
            static thread_local std::mt19937 gen(rd());
            std::normal_distribution<T> dist(mean, stddev);
            std::generate_n(data_.get(), total_size_, [&]() { return dist(gen); });
        }

        void xavier_init(size_type fan_in, size_type fan_out) {
            T limit = std::sqrt(T{ 6 } / static_cast<T>(fan_in + fan_out));
            static thread_local std::random_device rd;
            static thread_local std::mt19937 gen(rd());
            std::uniform_real_distribution<T> dist(-limit, limit);
            std::generate_n(data_.get(), total_size_, [&]() { return dist(gen); });
        }

    private:
        void reset() {
            data_.reset();
            shape_.fill(0);
            total_size_ = 0;
            dims_ = 0;
        }

        // Index computation
        template<typename... Indices>
        size_type compute_index(size_type first, Indices... rest) const {
            if constexpr (sizeof...(rest) == 0) {
                return first;
            }
            else {
                return first * shape_[sizeof...(rest)] + compute_index(rest...);
            }
        }

        // SIMD operations
        void simd_add(const Tensor& other) {
            if (total_size_ != other.total_size_) {
                throw std::invalid_argument("Size mismatch for addition");
            }

            if constexpr (std::is_same_v<T, float>) {
                simd_add_float(other);
            }
            else if constexpr (std::is_same_v<T, double>) {
                simd_add_double(other);
            }
            else {
                std::transform(data_.get(), data_.get() + total_size_,
                    other.data_.get(), data_.get(), std::plus<T>{});
            }
        }

        void simd_add_float(const Tensor& other) {
            const float* a = reinterpret_cast<const float*>(data_.get());
            const float* b = reinterpret_cast<const float*>(other.data_.get());
            float* result = reinterpret_cast<float*>(data_.get());

            size_type simd_size = total_size_ & ~7;

            for (size_type i = 0; i < simd_size; i += 8) {
                __m256 va = _mm256_load_ps(a + i);
                __m256 vb = _mm256_load_ps(b + i);
                __m256 vresult = _mm256_add_ps(va, vb);
                _mm256_store_ps(result + i, vresult);
            }
            for (size_type i = simd_size; i < total_size_; ++i) {
                result[i] = a[i] + b[i];
            }
        }

        void simd_add_double(const Tensor& other) {
            const double* a = reinterpret_cast<const double*>(data_.get());
            const double* b = reinterpret_cast<const double*>(other.data_.get());
            double* result = reinterpret_cast<double*>(data_.get());

            size_type simd_size = total_size_ & ~3;

            for (size_type i = 0; i < simd_size; i += 4) {
                __m256d va = _mm256_load_pd(a + i);
                __m256d vb = _mm256_load_pd(b + i);
                __m256d vresult = _mm256_add_pd(va, vb);
                _mm256_store_pd(result + i, vresult);
            }
            for (size_type i = simd_size; i < total_size_; ++i) {
                result[i] = a[i] + b[i];
            }
        }

        void simd_multiply(T scalar) {
            if constexpr (std::is_same_v<T, float>) {
                simd_multiply_float(scalar);
            }
            else if constexpr (std::is_same_v<T, double>) {
                simd_multiply_double(scalar);
            }
            else {
                std::transform(data_.get(), data_.get() + total_size_, data_.get(),
                    [scalar](T x) { return x * scalar; });
            }
        }

        void simd_multiply_float(float scalar) {
            float* data = reinterpret_cast<float*>(data_.get());
            __m256 vscalar = _mm256_set1_ps(scalar);

            size_type simd_size = total_size_ & ~7;

            for (size_type i = 0; i < simd_size; i += 8) {
                __m256 vdata = _mm256_load_ps(data + i);
                __m256 vresult = _mm256_mul_ps(vdata, vscalar);
                _mm256_store_ps(data + i, vresult);
            }
            for (size_type i = simd_size; i < total_size_; ++i) {
                data[i] *= scalar;
            }
        }

        void simd_multiply_double(double scalar) {
            double* data = reinterpret_cast<double*>(data_.get());
            __m256d vscalar = _mm256_set1_pd(scalar);

            size_type simd_size = total_size_ & ~3;

            for (size_type i = 0; i < simd_size; i += 4) {
                __m256d vdata = _mm256_load_pd(data + i);
                __m256d vresult = _mm256_mul_pd(vdata, vscalar);
                _mm256_store_pd(data + i, vresult);
            }
            for (size_type i = simd_size; i < total_size_; ++i) {
                data[i] *= scalar;
            }
        }

        void simd_apply(const std::function<T(T)>& func) {
            std::transform(data_.get(), data_.get() + total_size_, data_.get(), func);
        }

        void simd_matmul(const Tensor& other, Tensor& result) const {
            const size_type M = shape_[0];
            const size_type N = other.shape_[1];
            const size_type K = shape_[1];

            constexpr size_type block_size = 64;

            for (size_type i = 0; i < M; i += block_size) {
                for (size_type j = 0; j < N; j += block_size) {
                    for (size_type k = 0; k < K; k += block_size) {
                        size_type max_i = std::min(i + block_size, M);
                        size_type max_j = std::min(j + block_size, N);
                        size_type max_k = std::min(k + block_size, K);

                        for (size_type ii = i; ii < max_i; ++ii) {
                            for (size_type jj = j; jj < max_j; ++jj) {
                                T sum = T{ 0 };
                                for (size_type kk = k; kk < max_k; ++kk) {
                                    sum += data_[ii * K + kk] * other.data_[kk * N + jj];
                                }
                                result.data_[ii * N + jj] += sum;
                            }
                        }
                    }
                }
            }
        }
    };

    // ===== TensorBatch =====
    template<FloatingPoint T>
    class TensorBatch {
    private:
        std::vector<Tensor<T>> tensors_;
        size_t batch_size_;

    public:
        explicit TensorBatch(size_t batch_size) : batch_size_(batch_size) {
            tensors_.reserve(batch_size);
        }

        void add_tensor(Tensor<T> tensor) {
            if (tensors_.size() >= batch_size_) {
                throw std::runtime_error("Batch is full");
            }
            tensors_.push_back(std::move(tensor));
        }

        size_t size() const { return tensors_.size(); }
        Tensor<T>& operator[](size_t index) { return tensors_[index]; }
        const Tensor<T>& operator[](size_t index) const { return tensors_[index]; }

        auto begin() { return tensors_.begin(); }
        auto end() { return tensors_.end(); }
        auto cbegin() const { return tensors_.cbegin(); }
        auto cend() const { return tensors_.cend(); }
    };

}
