#pragma once

#include <cmath>
#include <algorithm>
#include "ExplorationStrategy.h"

/**
 * @brief Implements Boltzmann exploration strategy.
 *
 * This strategy uses a temperature parameter to control exploration.
 * Higher temperatures result in more exploration, while lower temperatures favor exploitation.
 */
class BoltzmannExploration : public ExplorationStrategy {
private:
    float temperature;   ///< Current temperature value
    float minTemp;       ///< Minimum temperature threshold
    float decay;         ///< Decay factor per step

public:
    /**
     * @brief Constructor to initialize Boltzmann exploration parameters.
     * @param startTemp Initial temperature value (default 1.0f)
     * @param minTemp Minimum allowable temperature (default 0.1f)
     * @param decay Rate at which temperature decays per step (default 0.995f)
     */
    BoltzmannExploration(float startTemp = 1.0f, float minTemp = 0.1f, float decay = 0.995f)
        : temperature(startTemp), minTemp(minTemp), decay(decay) {
    }

    /**
     * @brief Computes the current exploration rate based on the temperature and step.
     * @param step The current step of training or interaction.
     * @return Computed exploration rate.
     */
    float getExplorationRate(int step) const override {
        float temp = minTemp + (temperature - minTemp) * std::exp(-decay * static_cast<float>(step));
        return std::max(temp, minTemp);
    }

    /**
     * @brief Resets the internal state (e.g., temperature) to the initial value.
     */
    void reset() override {
        temperature = 1.0f;
    }
};
