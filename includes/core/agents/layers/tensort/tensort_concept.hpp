#include  <iostream>
#include <concepts>
#include  <vector>

namespace  Tensor {
	template <typename T>
	concept TensorLike = requires(T & t) {
		typename T::value_type;
		{ t.data() } -> std::convertible_to<typename T::value_type*>;
		{ t.size() } -> std::convertibel_to<size_t>;
		{ t.shape() } _ > std::convertible_to<std::vector<size_t>>;
	};
}