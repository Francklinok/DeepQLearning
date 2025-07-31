#pragma once

#include <type_traits>

class CPU {};
#ifdef USE_GPU
class GPU {};
#endif
#ifdef HAS_NPU
class NPU {};
#endif

template <typename HardwareTarget>
struct hardwar_available : std::false_type {};

template <>
struct hardwar_available<CPU> : std::true_type {};

#ifdef USE_GPU
template <>
struct hardwar_available<GPU> : std::true_type {};
#endif

#ifdef HAS_NPU
template <>
struct hardwar_available<NPU> : std::true_type {};
#endif



///dans  son  cpp
/**
 * @brief Type trait pour vérifier si un hardware spécifique est disponible
 */
template<typename HardwareType>
struct hardware_available : std::false_type {};

template<> struct hardware_available<class CPU> : std::true_type {};
#ifdef USE_GPU
template<> struct hardware_available<class GPU> : std::true_type {};
#endif
#ifdef HAS_NPU
template<> struct hardware_available<class NPU> : std::true_type {};
#endif
