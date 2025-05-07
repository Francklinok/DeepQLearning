#pragma once

/**
 * @brief Stores and manages metrics related to exploration behavior.
 *
 * @tparam T The numeric type used for metric values (default is float).
 */
template<typename T = float>
class ExplorationMetrics {
public:
    /**
     * @brief Default constructor initializing all metrics to zero.
     */
    ExplorationMetrics();

    // Getters

    /**
     * @brief Get the current value of exploration efficiency.
     * @return Exploration efficiency metric.
     */
    T getExplorationEfficiency() const;

    /**
     * @brief Get the current value of exploitation efficiency.
     * @return Exploitation efficiency metric.
     */
    T getExploitationEfficiency() const;

    /**
     * @brief Get the current value of entropy level.
     * @return Entropy level metric.
     */
    T getEntropyLevel() const;

    /**
     * @brief Get the current value of learning progress.
     * @return Learning progress metric.
     */
    T getLearningProgress() const;

    /**
     * @brief Get the current value of noise level.
     * @return Noise level metric.
     */
    T getNoiseLevel() const;

    // Setters

    /**
     * @brief Set the value of exploration efficiency.
     * @param value The new value to assign.
     */
    void setExplorationEfficiency(T value);

    /**
     * @brief Set the value of exploitation efficiency.
     * @param value The new value to assign.
     */
    void setExploitationEfficiency(T value);

    /**
     * @brief Set the value of entropy level.
     * @param value The new value to assign.
     */
    void setEntropyLevel(T value);

    /**
     * @brief Set the value of learning progress.
     * @param value The new value to assign.
     */
    void setLearningProgress(T value);

    /**
     * @brief Set the value of noise level.
     * @param value The new value to assign.
     */
    void setNoiseLevel(T value);

private:
    // Internal storage of metrics
    T explorationEfficiency;
    T exploitationEfficiency;
    T entropyLevel;
    T learningProgress;
    T noiseLevel;
};
