#pragma once

#include <memory>
#include <vector>
#include "ExplorationStrategy.hpp"
#include "EpsilonGreedy.hpp"
#include "NoisyExploration.hpp"
#include "EntropyBasedExploration.hpp"
#include "BoltzmannExploration.hpp"
#include "CompositeExploration.hpp"

/**
 * @brief Factory for creating different types of exploration strategies
 * @tparam T Data type used in computations (e.g., float or double)
 * @tparam HardwareTarget Hardware execution target (e.g., CPU, GPU)
 */
template<typename T = float, typename HardwareTarget = CPU>
class ExplorationFactory {
public:
    /**
     * @brief Creates an epsilon-greedy exploration strategy
     * @param start Initial epsilon value
     * @param end Minimum epsilon value
     * @param decay Decay factor for epsilon
     * @param minSteps Minimum steps before epsilon decay starts
     * @return Pointer to an epsilon-greedy strategy instance
     */
    static std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> createEpsilonGreedy(
        T start = static_cast<T>(1.0),
        T end = static_cast<T>(0.01),
        T decay = static_cast<T>(0.995),
        int64_t minSteps = 1000
    ) {
        return std::make_unique<EpsilonGreedy<T, HardwareTarget>>(start, end, decay, minSteps);
    }

    /**
     * @brief Creates a noise-based exploration strategy
     * @param scale Initial noise scale
     * @param decay Noise decay rate
     * @param adaptiveNoise Enable adaptive noise scaling
     * @return Pointer to a noise-based strategy instance
     */
    static std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> createNoisyExploration(
        T scale = static_cast<T>(0.5),
        T decay = static_cast<T>(0.99),
        bool adaptiveNoise = true
    ) {
        return std::make_unique<NoisyExploration<T, HardwareTarget>>(scale, decay, adaptiveNoise);
    }

    /**
     * @brief Creates an entropy-based exploration strategy
     * @param weight Initial entropy weight
     * @param minWeight Minimum entropy weight
     * @param decay Decay factor for the entropy weight
     * @param adaptiveFactor Dynamic adaptation factor
     * @return Pointer to an entropy-based strategy instance
     */
    static std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> createEntropyBasedExploration(
        T weight = static_cast<T>(0.01),
        T minWeight = static_cast<T>(0.001),
        T decay = static_cast<T>(0.995),
        T adaptiveFactor = static_cast<T>(1.2)
    ) {
        return std::make_unique<EntropyBasedExploration<T, HardwareTarget>>(
            weight, minWeight, decay, adaptiveFactor);
    }

    /**
     * @brief Creates a Boltzmann exploration strategy
     * @param initialTemp Initial temperature
     * @param minTemp Minimum temperature
     * @param coolingRate Rate at which temperature decreases
     * @param adaptiveCooling Enable adaptive cooling
     * @return Pointer to a Boltzmann strategy instance
     */
    static std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> createBoltzmannExploration(
        T initialTemp = static_cast<T>(1.0),
        T minTemp = static_cast<T>(0.01),
        T coolingRate = static_cast<T>(0.99),
        bool adaptiveCooling = true
    ) {
        return std::make_unique<BoltzmannExploration<T, HardwareTarget>>(
            initialTemp, minTemp, coolingRate, adaptiveCooling);
    }

    /**
     * @brief Creates a balanced composite strategy with equal weighting
     * @return Pointer to a composite strategy instance
     */
    static std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> createBalancedComposite() {
        auto composite = std::make_unique<CompositeExploration<T, HardwareTarget>>();
        composite->addStrategy(createEpsilonGreedy(), static_cast<T>(1.0));
        composite->addStrategy(createBoltzmannExploration(), static_cast<T>(1.0));
        composite->addStrategy(createNoisyExploration(), static_cast<T>(0.5));
        return composite;
    }

    /**
     * @brief Creates an adaptive composite strategy with dynamic weighting
     * @return Pointer to an adaptive composite strategy instance
     */
    static std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> createAdaptiveComposite() {
        // Fonction de combinaison qui favorise les valeurs extrêmes
        auto nonLinearCombinator = [](const std::vector<T>& rates) {
            if (rates.empty()) return static_cast<T>(0);
            T product = static_cast<T>(1.0);
            for (T rate : rates) {
                product *= (static_cast<T>(1.0) - rate);
            }
            return static_cast<T>(1.0) - product;
            };

        auto composite = std::make_unique<CompositeExploration<T, HardwareTarget>>(nonLinearCombinator);
        composite->addStrategy(createEpsilonGreedy(static_cast<T>(0.8), static_cast<T>(0.01), static_cast<T>(0.998), 2000), static_cast<T>(1.2));
        composite->addStrategy(createBoltzmannExploration(static_cast<T>(0.5), static_cast<T>(0.01), static_cast<T>(0.997), true), static_cast<T>(1.0));
        composite->addStrategy(createEntropyBasedExploration(), static_cast<T>(0.8));
        return composite;
    }
};
