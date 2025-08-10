#pragma once

#include <cmath>
#include <algorithm>
#include <string>
#include <type_traits>
#include <execution>
#include <iostream>
#include "activation.hpp"
#include "activation_concept.hpp"

template <typename T = float>
class ActivationFactory {
public:
    enum class Type {
        ReLU,
        LeakyReLU,
        ELU,
        Tanh
    };

    template <Type ActType>
    static auto create() {
        if constexpr (ActType == Type::ReLU) {
            return Activation::ReLUActivation<T>{};
        }
        else if constexpr (ActType == Type::LeakyReLU) {
            return Activation::LeakyReLUActivation<T>{};
        }
        else if constexpr (ActType == Type::ELU) {
            return Activation::ELUActivation<T>{};
        }
        else if constexpr (ActType == Type::Tanh) {
            return Activation::TanhActivation<T>{};
        }
        else {
            static_assert([] { return false; }(), "Unsupported activation type");
        }
    }
};
