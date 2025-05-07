#pragma once
#include "ExplorationStrategy.h"

/**
 * @brief Implements Boltzmann exploration strategy.
 *
 * This strategy uses a temperature parameter to control exploration.
 * Higher temperatures result in more exploration, while lower temperatures favor exploitation.
 */
class BoltzmannExploration : public ExplorationStrategy {
private:
    float temperature;  ///< Current temperature used for exploration.
    float minTemp;      ///< Minimum temperature threshold to prevent freezing.
    float decay;        ///< Decay rate applied to the temperature over time.

public:
    /**
     * @brief Constructor to initialize Boltzmann exploration parameters.
     *
     * @param startTemp Initial temperature value.
     * @param minTemp Minimum allowable temperature.
     * @param decay Rate at which temperature decays per step.
     */
    BoltzmannExploration(float startTemp = 1.0f, float minTemp = 0.1f, float decay = 0.995f);

    /**
     * @brief Computes the current exploration rate based on the temperature and step.
     *
     * @param step The current step of training or interaction.
     * @return Computed exploration rate.
     */
    float getExplorationRate(int step) const override;

    /**
     * @brief Resets the internal state (e.g., temperature) to the initial value.
     */
    void reset() override;
};
