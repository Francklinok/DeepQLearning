#pragma once  // Ensures this header is only included once during compilation

#include <iostream>     // For input/output operations (std::cout, std::cin, etc.)
#include <type_traits>  // For type traits like std::is_arithmetic_v and std::is_same_v
#include <concepts>     // For built-in C++20 concepts like std::floating_point

namespace dropout {

    // Concept: Numeric
    // Checks if T is an arithmetic type (integral or floating point) but NOT bool.
    // This helps avoid accidental use of bool in numeric computations.
    template <typename T>
    concept Numeric = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

    // Concept: FloatingPoint
    // Restricts T to be a floating-point type (float, double, long double, etc.).
    template <typename T>
    concept FloatingPoint = std::floating_point<T>;

} // namespace dropout
