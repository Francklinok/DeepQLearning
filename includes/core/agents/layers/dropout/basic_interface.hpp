#pragma once // Ensure this header is included only once during compilation

#include "dropout_concept.hpp" // For the Numeric concept definition

// Required standard headers
#include <vector>   // For std::vector
#include <string>   // For std::string
#include <span>     // For std::span (C++20)
#include <cstddef>  // For size_t

#ifdef TENSORFLOW_ENABLE
#include <tensorflow/core/framework/tensor.h>  // For tensorflow::Tensor
#include <tensorflow/cc/framework/ops.h>       // For tensorflow::Output, tensorflow::Scope, etc.
#endif

namespace dropout {

    // Base class for Dropout implementations
    // T is the numeric type (default: float) and must satisfy the Numeric concept.
    template <Numeric T = float>
    class DropoutBase {
    public:
        // Virtual destructor to allow proper cleanup of derived classes
        virtual ~DropoutBase() = default;

#ifdef TENSORFLOW_ENABLE
        // Apply dropout within a TensorFlow computation graph.
        // scope: TensorFlow scope for operation creation
        // input: Input tensor to apply dropout on
        // training: Whether dropout should be active (true) or bypassed (false)
        virtual tensorflow::Output apply(
            tensorflow::Scope& scope,
            tensorflow::Input input,
            bool training = true
        ) const = 0;
#endif

        // Apply dropout directly to a contiguous memory span of data (single-threaded).
        virtual void apply_dropout(std::span<T> data, bool training = true) = 0;

        // Apply dropout to a contiguous memory span of data in parallel.
        virtual void apply_dropout_parallel(std::span<T> data, bool training = true) = 0;

        // Forward pass: returns a new vector after applying dropout.
        // Default implementation could be overridden for efficiency.
        virtual std::vector<T> forward(const std::vector<T>& input, bool training = true);

        // Return the name/identifier of this dropout implementation.
        virtual std::string name() const = 0;

        // Getter for dropout rate (probability of dropping a value).
        virtual T get_dropout_rate() const = 0;

        // Setter for dropout rate.
        virtual void set_dropout_rate(T rate) = 0;

        // Enable or disable training mode (dropout active or not).
        virtual void set_training_mode(bool training) = 0;

        // Returns true if currently in training mode.
        virtual bool is_training() const = 0;
    };

} // namespace dropout
