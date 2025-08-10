#pragma once  // Ensure this header is included only once during compilation#pragma once

// Project-specific includes
#include "dropout_core.hpp"     // StandardDropout, GaussianDropout
#include "basic_interface.hpp"  // DropoutBase

// Standard library headers
#include <memory>      // std::unique_ptr, std::make_unique
#include <stdexcept>   // std::invalid_argument


namespace dropout {

    // Factory to create Dropout layers (Standard or Gaussian)
    template <Numeric T = float>
    class DropoutFactory {
    public:
        // Available dropout types
        enum class Type {
            Standard,
            Gaussian
        };

        // Compile-time creation of dropout layers
        template<Type DropType>
        static auto create(T dropout_rate = T(0.5)) {
            if constexpr (DropType == Type::Standard) {
                return std::make_unique<StandardDropout<T>>(dropout_rate);
            }
            else if constexpr (DropType == Type::Gaussian) {
                return std::make_unique<GaussianDropout<T>>(dropout_rate);
            }
        }

        // Runtime creation of dropout layers based on enum Type
        static std::unique_ptr<DropoutBase<T>> create_runtime(Type type, T dropout_rate = T(0.5)) {
            switch (type) {
            case Type::Standard:
                return std::make_unique<StandardDropout<T>>(dropout_rate);
            case Type::Gaussian:
                return std::make_unique<GaussianDropout<T>>(dropout_rate);
            default:
                throw std::invalid_argument("Unknown dropout type");
            }
        }
    };

    

} // namespace dropout
