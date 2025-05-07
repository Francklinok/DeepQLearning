
#include <ExplorationMetrics.h>

template<typename T>
ExplorationMetrics<T>::ExplorationMetrics()
    : explorationEfficiency(static_cast<T>(0.5)),
    exploitationEfficiency(static_cast<T>(0.5)),
    entropyLevel(static_cast<T>(1.0)),
    learningProgress(static_cast<T>(0)),
    noiseLevel(static_cast<T>(0)) {
}

template<typename T>
T ExplorationMetrics<T>::getExplorationEfficiency() const {
    return explorationEfficiency;
}

template<typename T>
T ExplorationMetrics<T>::getExploitationEfficiency() const {
    return exploitationEfficiency;
}

template<typename T>
T ExplorationMetrics<T>::getEntropyLevel() const {
    return entropyLevel;
}

template<typename T>
T ExplorationMetrics<T>::getLearningProgress() const {
    return learningProgress;
}

template<typename T>
T ExplorationMetrics<T>::getNoiseLevel() const {
    return noiseLevel;
}

template<typename T>
void ExplorationMetrics<T>::setExplorationEfficiency(T value) {
    explorationEfficiency = value;
}

template<typename T>
void ExplorationMetrics<T>::setExploitationEfficiency(T value) {
    exploitationEfficiency = value;
}

template<typename T>
void ExplorationMetrics<T>::setEntropyLevel(T value) {
    entropyLevel = value;
}

template<typename T>
void ExplorationMetrics<T>::setLearningProgress(T value) {
    learningProgress = value;
}

template<typename T>
void ExplorationMetrics<T>::setNoiseLevel(T value) {
    noiseLevel = value;
}