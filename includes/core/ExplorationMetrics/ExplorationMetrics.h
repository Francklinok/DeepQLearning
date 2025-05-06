#pragma once

template<typename T = float>
class ExplorationMetrics {
public:
	ExplorationMetrics();
	//getters
	T getExplorationEfficiency() const;
	T getExploitationEfficiency() const;
	T getEntropyLevel() const;
	T getLearningProgress() const;
	T getNoiseLevel() const;
	//setters

	void setExplorationEfficiency(T value);
	void setExploitationEfficiency(T value);
	void setEntropyLevel(T value);
	void setLearningProgress(T value);
	void setNoiseLevel(T value)

private:
	T explorationEfficiency;
	T exploitationEfficiency;
	T entropyLevel;
	T learningProgress;
	T noiseLevel;


};


