#include <iostream>
#include <concepts>
#include <vector>
#include <string>
#include <string_view>
#include <utility>

namespace Layer::Concept {
    // Concept for a neural network layer
    template<typename T>
    concept NeuralLayer = requires(T & t) {
        typename T::input_type;
        typename T::output_type;
        typename T::param_type;

        { t.forward(std::declval<const typename T::input_type&>()) }
        -> std::convertible_to<typename T::output_type>;

        { t.backward(std::declval<const typename T::output_type&>()) }
        -> std::convertible_to<typename T::input_type>;

        { t.get_params() }
        -> std::convertible_to<std::vector<typename T::param_type>>;

        { t.set_params(std::declval<const std::vector<typename T::param_type>&>()) }
        -> std::same_as<void>;

        { t.name() }
        -> std::convertible_to<std::string_view>; // cohérent avec BaseLayer
    };
}
