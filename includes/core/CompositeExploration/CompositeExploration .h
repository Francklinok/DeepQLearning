#pragma once

#include "ExplorationStrategy.hpp"
#include <vector>
#include <functional>
#include <numeric>

/**
 * @brief Composite strategy combining multiple exploration strategies
 * @tparam T Data type for computations
 * @tparam HardwareTarget Hardware target for execution
 */
template<typename T = float, typename HardwareTarget = CPU>
class CompositeExploration : public ExplorationStrategy<T, HardwareTarget> {
public:
    /**
     * @brief Type alias for the combination function of strategies
     */
    using CombinationFunction = std::function<T(const std::vector<T>&)>;

    /**
     * @brief Default constructor using average as the combination function
     */
    CompositeExploration();

    /**
     * @brief Constructor with a custom combination function
     * @param combFunc Combination function
     */
    explicit CompositeExploration(CombinationFunction combFunc);

    /**
     * @brief Adds a strategy with a specific weight
     * @param strategy Strategy to add
     * @param weight Weight of the strategy (1.0 by default)
     */
    void addStrategy(std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> strategy, T weight = static_cast<T>(1.0));

    /**
     * @brief Gets the exploration rate at the specified step
     * @param step Current step
     * @param context Optional execution context
     * @return Exploration rate
     */
    T getExplorationRate(int64_t step, const typename ExplorationStrategy<T, HardwareTarget>::context_type* context = nullptr) const override;

    /**
     * @brief Resets all strategies
     */
    void reset() override;

    /**
     * @brief Adjusts strategy parameters based on metrics
     * @param metrics Exploration metrics
     */
    void adaptToMetrics(const typename ExplorationStrategy<T, HardwareTarget>::metrics_type& metrics) override;

    /**
     * @brief Clones the composite strategy
     * @return Unique pointer to a new instance
     */
    std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> clone() const override;

private:
    // Strategy-weight pair
    struct WeightedStrategy {
        std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> strategy;
        T weight;
    };

    std::vector<WeightedStrategy> strategies;  // List of strategies with weights
    CombinationFunction combinator;           // Combination function
};

#include "CompositeExploration.inl"
