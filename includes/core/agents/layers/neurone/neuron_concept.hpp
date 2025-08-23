#include <iostream>
#include <concepts>
#include <vector>
#include <string>
#include <utility>

namespace Neurone {
    //concept  for  a  neural  network  layer
    template<typename T>
    concept NeuraLayer = requires(T & t) {
        typename T::input_type;
        typename T::output_type;
        typename T::param_type;

        { t.forward(std::declval<typename T::input_type>()) }
        -> std::convertible_to<typename T::output_type>;

        { t.backward(std::declval<typename T::output_type>()) }
        -> std::convertible_to<typename T::input_type>;

        { t.get_params() }
        -> std::convertible_to<std::vector<typename T::param_type>>;

        { t.set_params(std::declval<std::vector<typename T::param_type>>()) }
        -> std::same_as<void>;

        { t.name() }
        -> std::convertible_to<std::string>;
    };
}
