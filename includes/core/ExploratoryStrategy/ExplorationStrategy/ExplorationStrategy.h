#pragma once

#include <memory>
#include <cstdint>
#include <type_traits>
#include "ExplorationMetrics.h"
#include "ExecutionContext.h"
#include "HardwareTraits.h"

template <typename T = float, typename HardwareTarget = CPU>
class ExplorationStrategy {
public:
    using value_type = T;
    using metrics_type = ExplorationMetrics<T>;
    using context_type = ExecutionContext<T>;

    static_assert(std::is_floating_point_v<T>, "Le type doit être à virgule flottante");
    static_assert(hardwar_available<HardwareTarget>::value, "Hardware target not available");

    virtual ~ExplorationStrategy() = default;

    virtual T getExploratoryRate(int64_t step, const context_type* context = nullptr) const = 0;
    virtual void reset() = 0;

    virtual void adapToMetrics(const metrics_type& metrics) {}

    virtual std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> clone() const = 0;

protected:
#ifdef USE_GPU
    __device__ virtual T getExplorationRateDevice(int64_t step) const;
#endif
};