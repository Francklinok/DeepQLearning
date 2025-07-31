#pragma
#include <iostream>
#include  <type_traits>
#include  <cstdint>


namespace normalizaton {
	//security concept  
	template <typename T>
	concept FloatingPoint = std::floating_point<T>;

	template <typename T>
	concept  SimdCompatible = std::is_same_v<T, float> || std::is_same_v<T, double>;

	//  normalize  type
	enum  class NormalizationType : uint8_t {
		Batch = 0,
		Layer = 1,
		Instance = 2,
		Group = 3
	};

	//optimisation constante
	namespace constants {
		static  constexpr size_t CACHE_LINE_SIZE = 64;
		static constexpr size_t SIMD_WIDTH_F32 = 8;
		static constexpr size_t SIMD_WIDTH_F64 = 4;
		static constexpr size_t MAX_NORMALIZATION_TYPES = 8;

	}

}