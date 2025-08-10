#pragma once
#include <iostream>
#include <type_traits>
#include <concepts>
#include <string>

namespace Activation {

    template <typename T>
    concept Numeric = std::is_arithmetic_v<T>;

    template <typename T>
    concept FloatingPoint = std::is_floating_point_v<T>;

    template <typename Func, typename T>
    concept ActivationFunction = requires(Func & f, T x) {   // correction : utiliser 'x' défini ici
        { f.activate(x) } -> std::convertible_to<T>;
        { f.derivative(x) } -> std::convertible_to<T>;
        { f.name() } -> std::convertible_to<std::string>;
    };

}
