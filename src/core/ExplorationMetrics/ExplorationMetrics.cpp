#include <ExplorationMetrics.h>

ExplorationMetrics::ExplorationMetrics()
	:explorationEfficiency(static_cast<T>(0.5)),
	exploitationEfficiency(static_cast<T>(0.5)),
	entropyLevel(static_cast<T>(1.0)),
    learningProgress(static_cast<T>(0)),
	noiseLevel(static_cast<T>(0)) {}

T ExplorationMetrics::getExplorationEfficiency() const {
	return explorationEfficiency; 
}

T ExplorationMetrics::getExploitationEfficiency() const {
	return exploitationEfficiency; 
}
T ExplorationMetrics::getEntropyLevel() const {
	return entropyLevel; 
}
T ExplorationMetrics::getLearningProgress() const {
	return learningProgress;
}

T ExplorationMetrics::getNoiseLevel() const {
	return noiseLevel;
}

void ExplorationMetrics::setExplorationEfficiency(T value) { 
	explorationEfficiency = value; 
}
void ExplorationMetrics::setExploitationEfficiency(T value) { 
	exploitationEfficiency = value; 
}
void ExplorationMetrics::setEntropyLevel(T value) { 
	entropyLevel = value;
}

void ExplorationMetrics::setLearningProgress(T value) {
	learningProgress = value;
}
void ExplorationMetrics::setNoiseLevel(T value) {
	noiseLevel = value; 
}



