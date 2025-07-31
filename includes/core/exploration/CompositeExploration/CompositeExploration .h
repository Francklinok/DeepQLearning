#pragma once
#include "ExplorationStrategy.hpp"
#include <memory>
#include <vector>
#include <functional>
#include <numeric>
#include <cmath>

/**
 * @brief Composite strategy combining multiple exploration strategies
 * @tparam T Data type for computations
 * @tparam HardwareTarget Hardware target for execution
 */
template<typename T, typename HardwareTarget>
class CompositeExploration : public ExplorationStrategy<T, HardwareTarget> {
public:
    /**
     * @brief Type alias for the combination function of strategies
     */
    using CombinationFunction = std::function<T(const std::vector<T>&)>;

    /**
     * @brief Default constructor using average as the combination function
     */
    CompositeExploration()
        : combinator([](const std::vector<T>& rates) {
        if (rates.empty()) return static_cast<T>(0);
        return std::accumulate(rates.begin(), rates.end(), static_cast<T>(0)) / static_cast<T>(rates.size());
            }) {
    }

    /**
     * @brief Constructor with a custom combination function
     * @param combFunc Combination function
     */
    explicit CompositeExploration(CombinationFunction combFunc)
        : combinator(std::move(combFunc)) {
    }

    /**
     * @brief Adds a strategy with a specific weight
     * @param strategy Strategy to add
     * @param weight Weight of the strategy (1.0 by default)
     */
    void addStrategy(std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> strategy, T weight = static_cast<T>(1.0)) {
        strategies.push_back({ std::move(strategy), weight });
    }

    /**
     * @brief Gets the exploration rate at the specified step
     * @param step Current step
     * @param context Optional execution context
     * @return Exploration rate
     */
    T getExplorationRate(int64_t step, const typename ExplorationStrategy<T, HardwareTarget>::context_type* context = nullptr) const override {
        // Si aucune stratégie, retourner 0
        if (strategies.empty()) return static_cast<T>(0);

        // Utiliser le cache si disponible
        if (context) {
            T cachedValue = context->getCachedValue(step, "composite");
            if (!std::isnan(cachedValue)) {
                return cachedValue;
            }
        }

        // Calculer les taux pour chaque stratégie
        std::vector<T> rates;
        rates.reserve(strategies.size());
        T totalWeight = static_cast<T>(0);

        for (const auto& [strategy, weight] : strategies) {
            rates.push_back(strategy->getExplorationRate(step, context) * weight);
            totalWeight += weight;
        }

        // Normaliser les poids si nécessaire
        if (totalWeight > 0 && totalWeight != static_cast<T>(1.0)) {
            for (auto& rate : rates) {
                rate /= totalWeight;
            }
        }

        // Combiner les taux selon la fonction de combinaison
        T result = combinator(rates);

        // Mettre en cache le résultat
        if (context) {
            context->setCachedValue(step, "composite", result);
        }

        return result;
    }

    /**
     * @brief Resets all strategies
     */
    void reset() override {
        for (auto& [strategy, _] : strategies) {
            strategy->reset();
        }
    }

    /**
     * @brief Adjusts strategy parameters based on metrics
     * @param metrics Exploration metrics
     */
    void adaptToMetrics(const typename ExplorationStrategy<T, HardwareTarget>::metrics_type& metrics) override {
        for (auto& [strategy, _] : strategies) {
            strategy->adaptToMetrics(metrics);
        }
    }

    /**
     * @brief Clones the composite strategy
     * @return Unique pointer to a new instance
     */
    std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> clone() const override {
        auto cloned = std::make_unique<CompositeExploration<T, HardwareTarget>>(combinator);
        for (const auto& [strategy, weight] : strategies) {
            cloned->addStrategy(strategy->clone(), weight);
        }
        return cloned;
    }

private:
    // Strategy-weight pair
    struct WeightedStrategy {
        std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> strategy;
        T weight;
    };

    std::vector<WeightedStrategy> strategies;  // List of strategies with weights
    CombinationFunction combinator;           // Combination function
};