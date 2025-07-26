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
     * @brief Default constructor initializing all metrics to default values.
     */
    ExplorationMetrics()
        : explorationEfficiency(static_cast<T>(0.5)),
        exploitationEfficiency(static_cast<T>(0.5)),
        entropyLevel(static_cast<T>(1.0)),
        learningProgress(static_cast<T>(0)),
        noiseLevel(static_cast<T>(0)) {
    }

    // === Getters ===
    inline T getExplorationEfficiency() const { return explorationEfficiency; }
    inline T getExploitationEfficiency() const { return exploitationEfficiency; }
    inline T getEntropyLevel() const { return entropyLevel; }
    inline T getLearningProgress() const { return learningProgress; }
    inline T getNoiseLevel() const { return noiseLevel; }

    // === Setters ===
    inline void setExplorationEfficiency(T value) { explorationEfficiency = value; }
    inline void setExploitationEfficiency(T value) { exploitationEfficiency = value; }
    inline void setEntropyLevel(T value) { entropyLevel = value; }
    inline void setLearningProgress(T value) { learningProgress = value; }
    inline void setNoiseLevel(T value) { noiseLevel = value; }

private:
    // Internal storage of metrics
    T explorationEfficiency;
    T exploitationEfficiency;
    T entropyLevel;
    T learningProgress;
    T noiseLevel;
};
